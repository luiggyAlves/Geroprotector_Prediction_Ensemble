"""Machinery for EX_027.

EX_027 runs the whole EX_024 pipeline twice -- once under a Bemis-Murcko scaffold protocol
and once under the original random stratified protocol -- on top of the corrections
introduced in EX_026: scaffold splitting, in-fold greedy subset selection, in-fold threshold
selection with threshold-free screening metrics, dependency-aware statistics (group and
paired bootstrap, Nadeau-Bengio), a label-conflict ablation, a chemical standardization
audit, and a locked external test set.

Sections 1-8 are the EX_026 module verbatim, so both notebooks compute the same quantities
from the same code and the same checkpoints. Section 9 is what EX_027 adds, and all of it is
derived from the per-fold records EX_026 already wrote to disk, so nothing in the nested
protocol is refit:

* every individual representation gets its OWN in-fold threshold, instead of inheriting the
  ensemble's -- the ranking of the 20 representations is the central claim of the paper, and
  in EX_024 it was read at a single arbitrary point of each model's score distribution;
* voting over all representations is rebuilt with weights taken from the INNER out-of-fold
  AUC rather than the outer test AUC, which is where EX_024 leaked the test fold into the
  ensemble weights;
* an oracle-selection contrast measures, in the same folds, how much choosing the subset
  against the evaluation data inflates the result;
* learning curves subsample whole scaffold groups under the scaffold protocol, because
  subsampling compounds would let a congeneric series re-enter the smaller training sets and
  turn the curve into a picture of memorization;
* diversity is measured on out-of-fold predictions taken at each member's own threshold.

Nothing in this module touches the external test set. EX_027 reuses the configuration frozen
by EX_026 and the numbers it produced; the lock-box is not opened again.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict

import joblib
import numpy as np
import pandas as pd

from scipy import stats

from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# --------------------------------------------------------------------------------------
# 1. Chemistry: standardization and Bemis-Murcko scaffolds
# --------------------------------------------------------------------------------------
#
# RDKit is imported lazily: the modelling sections of EX_026 run without it, and only the
# chemistry audit (section 2 of the notebook) needs it.

_STD_CACHE: dict = {}


def _rdkit():
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")        # standardizing 5,790 molecules is otherwise very noisy
    return Chem, rdMolStandardize, MurckoScaffold


def standardization_levels(smiles):
    """Returns one identifier per standardization level for a single SMILES.

    The levels are cumulative, so comparing the number of distinct values between two
    consecutive levels says exactly how many duplicates that one operation exposed:

    ``L0_canonical``   RDKit canonical SMILES (what the current dedup of the dataset uses)
    ``L1_cleanup``     + ``Cleanup`` (sanitization, metal disconnection, normalization)
    ``L2_parent``      + ``FragmentParent`` (removes salt/solvate counter-fragments)
    ``L3_uncharged``   + ``Uncharger`` (neutralizes the charge state)
    ``L4_tautomer``    + canonical tautomer
    ``L5_inchikey``    InChIKey of the L4 molecule (stereochemistry-sensitive)
    ``L6_skeleton``    first block of that InChIKey (constitution only: ignores stereochemistry)
    """
    if smiles in _STD_CACHE:
        return _STD_CACHE[smiles]

    Chem, rdMolStandardize, _ = _rdkit()
    out = {k: None for k in ("L0_canonical", "L1_cleanup", "L2_parent", "L3_uncharged",
                             "L4_tautomer", "L5_inchikey", "L6_skeleton")}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        _STD_CACHE[smiles] = out
        return out

    out["L0_canonical"] = Chem.MolToSmiles(mol)
    mol = rdMolStandardize.Cleanup(mol)
    out["L1_cleanup"] = Chem.MolToSmiles(mol)
    mol = rdMolStandardize.FragmentParent(mol)
    out["L2_parent"] = Chem.MolToSmiles(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    out["L3_uncharged"] = Chem.MolToSmiles(mol)
    try:
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    except Exception:
        pass                              # a few large molecules exceed the enumerator limits
    out["L4_tautomer"] = Chem.MolToSmiles(mol)
    key = Chem.MolToInchiKey(mol)
    out["L5_inchikey"] = key or None
    out["L6_skeleton"] = key.split("-")[0] if key else None
    _STD_CACHE[smiles] = out
    return out


def murcko_scaffold(smiles, include_chirality=False):
    """Bemis-Murcko scaffold of the standardized (desalted, neutral, canonical-tautomer) molecule.

    Returns ``None`` for molecules RDKit cannot parse and ``""`` for acyclic molecules,
    whose Murcko scaffold is empty by definition.
    """
    Chem, _, MurckoScaffold = _rdkit()
    parent = standardization_levels(smiles)["L4_tautomer"]
    if parent is None:
        return None
    mol = Chem.MolFromSmiles(parent)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)


def chemistry_table(smiles_list):
    """Standardization levels, scaffold and grouping key for every SMILES, in input order.

    ``group`` is what the scaffold splitter receives. Molecules that share a Bemis-Murcko
    scaffold share a group. Acyclic molecules have no scaffold, so each one becomes its own
    group keyed by its standardized InChIKey: lumping all of them together would create a
    single huge group and distort the folds, while splitting them per row would separate
    exact duplicates. Rows RDKit cannot parse keep a per-row group.
    """
    rows = []
    for i, smi in enumerate(smiles_list):
        levels = standardization_levels(smi)
        scaffold = murcko_scaffold(smi)
        if scaffold:
            group = f"scaffold:{scaffold}"
        elif levels["L5_inchikey"]:
            group = f"acyclic:{levels['L5_inchikey']}"
        else:
            group = f"unparsed:{i}"
        rows.append({"index": i, "smiles": smi, "scaffold": scaffold, "group": group, **levels})
    return pd.DataFrame(rows)


def duplicate_audit(chem):
    """Counts how many of the rows collapse at each standardization level.

    ``n_unique`` is the number of distinct identifiers, ``n_redundant`` the number of rows
    that are a repeat of an earlier row, and ``newly_exposed`` the extra collapses this level
    produces relative to the previous one.
    """
    levels = ["L0_canonical", "L1_cleanup", "L2_parent", "L3_uncharged",
              "L4_tautomer", "L5_inchikey", "L6_skeleton"]
    rows, previous = [], None
    for level in levels:
        values = chem[level].dropna()
        n_unique = values.nunique()
        n_redundant = len(values) - n_unique
        rows.append({"level": level, "n_rows_resolved": len(values), "n_unique": n_unique,
                     "n_redundant": n_redundant,
                     "newly_exposed": 0 if previous is None else n_redundant - previous})
        previous = n_redundant
    return pd.DataFrame(rows)


def duplicate_pairs(chem, level="L5_inchikey"):
    """Row-index groups that collapse to the same identifier at ``level`` (size >= 2)."""
    resolved = chem.dropna(subset=[level])
    counts = resolved.groupby(level)["index"].apply(list)
    return {key: idx for key, idx in counts.items() if len(idx) > 1}


# --------------------------------------------------------------------------------------
# 2. Splits
# --------------------------------------------------------------------------------------


def repeated_random_folds(y, n_splits=5, seeds=(42, 43, 44)):
    """The EX_024 protocol: repeated stratified K-fold over individual compounds."""
    folds = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds += [(seed, k, tr, te) for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y))]
    return folds


def repeated_scaffold_folds(y, groups, n_splits=5, seeds=(42, 43, 44)):
    """Repeated stratified group K-fold: a whole Bemis-Murcko scaffold stays on one side.

    ``StratifiedGroupKFold`` keeps the group constraint hard and the class ratio of each fold
    as close to the global ratio as the groups allow. The seeds only reshuffle the groups, so
    every repetition still predicts each compound exactly once.
    """
    folds = []
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds += [(seed, k, tr, te)
                  for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups=groups))]
    return folds


def split_leakage_report(folds, chem, level="L5_inchikey"):
    """How many standardized-duplicate pairs end up with one member in train and one in test.

    Each such pair is direct leakage: the model sees the same substance in both roles. The
    count is what tells you how much of the optimism of a protocol comes from duplicates
    rather than from congeneric series.
    """
    groups_of_duplicates = duplicate_pairs(chem, level=level)
    rows = []
    for seed, k, tr, te in folds:
        in_test = np.zeros(len(chem), dtype=bool)
        in_test[te] = True
        separated = sum(1 for idx in groups_of_duplicates.values()
                        if in_test[idx].any() and (~in_test[idx]).any())
        rows.append({"seed": seed, "fold": k, "separated_duplicate_groups": separated})
    return pd.DataFrame(rows)


def fold_composition(folds, y, groups):
    """Per-fold sizes, class balance and how many test scaffolds are unseen in training."""
    groups = np.asarray(groups)
    rows = []
    for seed, k, tr, te in folds:
        unseen = len(set(groups[te]) - set(groups[tr]))
        rows.append({"seed": seed, "fold": k, "n_train": len(tr), "n_test": len(te),
                     "test_prevalence": float(y[te].mean()),
                     "n_test_groups": len(set(groups[te])),
                     "n_test_groups_unseen_in_train": unseen})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 3. Metrics
# --------------------------------------------------------------------------------------

THRESHOLD_FREE = ["roc_auc", "average_precision", "bedroc_20", "ef_5pct", "ef_100", "ef_200",
                  "precision_at_50", "precision_at_100", "precision_at_200"]
THRESHOLD_DEPENDENT = ["f1", "precision", "recall", "specificity", "balanced_accuracy",
                       "accuracy", "matthews_corrcoef"]

METRIC_LABELS = {
    "roc_auc": "ROC-AUC", "average_precision": "PR-AUC", "bedroc_20": "BEDROC (alpha=20)",
    "ef_5pct": "EF5%", "ef_100": "EF@100", "ef_200": "EF@200",
    "precision_at_50": "Precision@50", "precision_at_100": "Precision@100",
    "precision_at_200": "Precision@200", "f1": "F1", "precision": "Precision",
    "recall": "Recall", "specificity": "Specificity", "balanced_accuracy": "Balanced accuracy",
    "accuracy": "Accuracy", "matthews_corrcoef": "MCC",
}


def precision_at_k(y_true, scores, k):
    """Fraction of true positives among the k highest-scoring compounds.

    This is the screening question: of the k compounds sent to the bench, how many are hits.
    Ties are broken by a stable sort, so the value is reproducible.
    """
    k = min(k, len(y_true))
    top = np.argsort(-np.asarray(scores), kind="mergesort")[:k]
    return float(np.asarray(y_true)[top].mean())


def enrichment_factor(y_true, scores, k):
    """Precision@k divided by the prevalence: how many times better than picking at random."""
    prevalence = float(np.mean(y_true))
    if prevalence == 0:
        return np.nan
    return precision_at_k(y_true, scores, k) / prevalence


def bedroc(y_true, scores, alpha=20.0):
    """Boltzmann-enhanced discrimination of ROC (Truchon & Bayly 2007).

    An exponential weight on the rank makes the top of the list dominate the score, which is
    the only part of a virtual-screening ranking anyone inspects. alpha=20 puts roughly 80% of
    the weight on the top 8% of the ranking. The value is on [0, 1]; 0.5 is random.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == n:
        return np.nan

    order = np.argsort(-scores, kind="mergesort")
    ranks = np.flatnonzero(y_true[order] == 1) + 1
    ratio = n_pos / n

    rie_numerator = np.sum(np.exp(-alpha * ranks / n))
    rie_denominator = ratio * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)
    rie = rie_numerator / rie_denominator

    scale = (ratio * np.sinh(alpha / 2)
             / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * ratio)))
    offset = 1.0 / (1 - np.exp(alpha * (1 - ratio)))
    return float(rie * scale + offset)


def all_metrics(y_true, scores, threshold):
    """Every reported metric for one score vector; threshold-free metrics ignore ``threshold``."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "roc_auc": roc_auc_score(y_true, scores) if 0 < y_true.sum() < len(y_true) else np.nan,
        "average_precision": average_precision_score(y_true, scores),
        "bedroc_20": bedroc(y_true, scores, alpha=20.0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "threshold": float(threshold),
    }
    for k in (50, 100, 200):
        out[f"precision_at_{k}"] = precision_at_k(y_true, scores, k)
    # EF5% is the enrichment in the top 5% of the ranked list, the usual early-recognition cut
    # in virtual screening. Being a fraction rather than a fixed count, it means the same thing
    # on the 4,233-compound primary set, on a bootstrap replicate and on a smaller external set.
    out["ef_5pct"] = enrichment_factor(y_true, scores, max(1, int(round(0.05 * len(y_true)))))
    for k in (100, 200):
        out[f"ef_{k}"] = enrichment_factor(y_true, scores, k)
    return out


# --------------------------------------------------------------------------------------
# 4. Threshold policies
# --------------------------------------------------------------------------------------

THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


def choose_threshold(y_true, scores, policy="max_f1", min_recall=0.5, grid=THRESHOLD_GRID):
    """Picks a decision threshold from a declared policy. Call this on training data only.

    ``max_f1``                  the threshold with the highest F1.
    ``max_precision_at_recall`` the highest precision among thresholds reaching
                                ``min_recall``; closer to how a screening campaign is run,
                                where a recall floor is set and precision is maximized under it.
    ``fixed_0.5``               the sklearn default, kept so EX_026 can quantify what it costs.
    """
    if policy.startswith("fixed"):
        return float(policy.split("_")[-1])

    y_true = np.asarray(y_true)
    best_threshold, best_key = 0.5, None
    for t in grid:
        y_pred = (np.asarray(scores) >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        if policy == "max_f1":
            key = (f1, precision)
        elif policy == "max_precision_at_recall":
            key = (recall >= min_recall, precision, recall)
        else:
            raise ValueError(f"unknown policy: {policy}")
        if best_key is None or key > best_key:
            best_key, best_threshold = key, float(t)
    return best_threshold


def threshold_sweep(y_true, scores, grid=THRESHOLD_GRID):
    """Every threshold-dependent metric over the whole grid: the curve 0.5 sits on."""
    rows = []
    for t in grid:
        m = all_metrics(y_true, scores, t)
        rows.append({"threshold": float(t), **{k: m[k] for k in THRESHOLD_DEPENDENT}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 5. Statistics that respect the dependence structure
# --------------------------------------------------------------------------------------


def group_bootstrap_indices(groups, n_boot=2000, seed=RANDOM_STATE):
    """Resamples whole groups (scaffolds) with replacement and yields row indices.

    Compounds that share a scaffold are not independent observations, so resampling rows
    individually treats a congeneric series as many independent draws and shrinks the interval.
    Resampling the group keeps the unit of independence correct.
    """
    groups = np.asarray(groups)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        drawn = rng.choice(unique, size=unique.size, replace=True)
        yield np.concatenate([members[g] for g in drawn])


def group_bootstrap_ci(y_true, scores, groups, threshold, metrics=None, n_boot=2000,
                       seed=RANDOM_STATE, alpha=0.05):
    """Percentile confidence intervals for one model, resampling by group."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    draws = []
    for idx in group_bootstrap_indices(groups, n_boot=n_boot, seed=seed):
        if y_true[idx].sum() == 0 or y_true[idx].sum() == len(idx):
            continue                                   # a replicate with one class has no AUC
        draws.append(all_metrics(y_true[idx], scores[idx], threshold))
    draws = pd.DataFrame(draws)
    metrics = metrics or [m for m in draws.columns if m != "threshold"]
    return pd.DataFrame({
        "metric": metrics,
        "point": [all_metrics(y_true, scores, threshold)[m] for m in metrics],
        "ci_low": [draws[m].quantile(alpha / 2) for m in metrics],
        "ci_high": [draws[m].quantile(1 - alpha / 2) for m in metrics],
        "n_replicates": len(draws),
    })


def paired_group_bootstrap(y_true, scores_a, scores_b, groups, threshold_a, threshold_b,
                           metrics=None, n_boot=2000, seed=RANDOM_STATE, alpha=0.05):
    """Confidence interval of the difference A - B under the same resampling for both models.

    Two overlapping one-model intervals do not imply the difference is not significant when
    the models are correlated, and models scored on the same compounds always are. The paired
    interval is the one that answers "does A beat B".
    """
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    diffs = []
    for idx in group_bootstrap_indices(groups, n_boot=n_boot, seed=seed):
        if y_true[idx].sum() == 0 or y_true[idx].sum() == len(idx):
            continue
        ma = all_metrics(y_true[idx], scores_a[idx], threshold_a)
        mb = all_metrics(y_true[idx], scores_b[idx], threshold_b)
        diffs.append({k: ma[k] - mb[k] for k in ma if k != "threshold"})
    diffs = pd.DataFrame(diffs)
    metrics = metrics or list(diffs.columns)
    point_a = all_metrics(y_true, scores_a, threshold_a)
    point_b = all_metrics(y_true, scores_b, threshold_b)
    return pd.DataFrame({
        "metric": metrics,
        "A": [point_a[m] for m in metrics],
        "B": [point_b[m] for m in metrics],
        "difference": [point_a[m] - point_b[m] for m in metrics],
        "ci_low": [diffs[m].quantile(alpha / 2) for m in metrics],
        "ci_high": [diffs[m].quantile(1 - alpha / 2) for m in metrics],
        "p_two_sided": [float(2 * min((diffs[m] <= 0).mean(), (diffs[m] >= 0).mean()))
                        for m in metrics],
        "excludes_zero": [bool(diffs[m].quantile(alpha / 2) > 0 or
                               diffs[m].quantile(1 - alpha / 2) < 0) for m in metrics],
        "n_replicates": len(diffs),
    })


def corrected_resampled_ttest(scores_a, scores_b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated K-fold differences.

    Repeated K-fold folds are not independent: their training sets overlap heavily. The usual
    paired t-test over fold differences ignores that and returns anti-conservative p-values.
    The correction replaces the 1/n variance factor with ``1/n + n_test/n_train``.
    """
    d = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    n = d.size
    variance = d.var(ddof=1)
    if variance == 0:
        return {"mean_difference": float(d.mean()), "t": np.nan, "p_value": np.nan,
                "df": n - 1, "naive_p_value": np.nan}
    factor = 1.0 / n + n_test / n_train
    t_corrected = d.mean() / np.sqrt(factor * variance)
    t_naive = d.mean() / np.sqrt(variance / n)
    return {
        "mean_difference": float(d.mean()),
        "sd_difference": float(np.sqrt(variance)),
        "t": float(t_corrected),
        "p_value": float(2 * stats.t.sf(abs(t_corrected), df=n - 1)),
        "df": n - 1,
        "variance_factor": float(factor),
        "naive_p_value": float(2 * stats.t.sf(abs(t_naive), df=n - 1)),
    }


def bootstrap_optimal_threshold(y_true, scores, groups, objective="f1", n_boot=2000,
                                seed=RANDOM_STATE, alpha=0.05, grid=THRESHOLD_GRID):
    """Confidence interval of the threshold that maximizes ``objective``.

    If 0.5 falls outside this interval, the sklearn default is empirically rejected for this
    model and this dataset, which is a stronger statement than "0.5 is arbitrary".
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    def best(y, s):
        values = [f1_score(y, (s >= t).astype(int), zero_division=0) if objective == "f1"
                  else matthews_corrcoef(y, (s >= t).astype(int)) for t in grid]
        return float(grid[int(np.argmax(values))]), float(np.max(values))

    point_threshold, point_value = best(y_true, scores)
    draws = []
    for idx in group_bootstrap_indices(groups, n_boot=n_boot, seed=seed):
        if y_true[idx].sum() == 0:
            continue
        draws.append(best(y_true[idx], scores[idx])[0])
    draws = np.asarray(draws)
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "objective": objective,
        "optimal_threshold": point_threshold,
        "optimal_value": point_value,
        "ci_low": float(low),
        "ci_high": float(high),
        "contains_0.5": bool(low <= 0.5 <= high),
        "n_replicates": int(draws.size),
        "draws": draws,
    }


# --------------------------------------------------------------------------------------
# 6. Models, checkpoints and the nested procedure
# --------------------------------------------------------------------------------------


def build_pipeline(binary=False, C=1.0, random_state=RANDOM_STATE):
    """The EX_024 pipeline: L2 logistic regression, with near-constant bits dropped first
    for binary fingerprints."""
    clf = LogisticRegression(solver="lbfgs", max_iter=5000, class_weight="balanced",
                             C=C, random_state=random_state)
    steps = [("var", VarianceThreshold(threshold=0.01))] if binary else []
    steps += [("scaler", StandardScaler()), ("clf", clf)]
    return Pipeline(steps)


class Checkpoints:
    """Atomic joblib checkpoints, so an interrupted run resumes instead of restarting."""

    def __init__(self, directory):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def path(self, key):
        return os.path.join(self.directory, f"{key}.joblib")

    def has(self, key):
        return os.path.exists(self.path(key))

    def save(self, key, value):
        path = self.path(key)
        joblib.dump(value, path + ".tmp")
        os.replace(path + ".tmp", path)
        return path

    def load(self, key):
        return joblib.load(self.path(key))

    def run_or_load(self, key, fn, verbose=True):
        if self.has(key):
            if verbose:
                print(f"[checkpoint] loaded {key}")
            return self.load(key)
        started = time.time()
        value = fn()
        self.save(key, value)
        if verbose:
            print(f"[checkpoint] computed {key} in {time.time() - started:.1f}s")
        return value


def inner_oof_and_C(X, y, inner_folds, C_grid, binary=False, n_jobs=1):
    """Chooses C and produces inner out-of-fold probabilities in a single pass.

    One fit per (inner fold, C) gives both: the mean inner ROC-AUC selects C, and the
    out-of-fold column of that same C is kept. Every quantity here comes from the outer
    training part only, which is what makes the outer fold an untouched test set.

    This is also what makes a full nested protocol affordable. Soft voting is an average of
    probability columns, so once this matrix exists the greedy search costs no refits at all.
    """
    from joblib import Parallel, delayed

    C_grid = np.asarray(C_grid, dtype=float)
    n = len(y)

    def one_fold(tr, va):
        columns = {}
        for C in C_grid:
            model = build_pipeline(binary=binary, C=C).fit(X[tr], y[tr])
            columns[C] = model.predict_proba(X[va])[:, 1]
        return va, columns

    results = Parallel(n_jobs=n_jobs)(delayed(one_fold)(tr, va) for tr, va in inner_folds)

    oof = {C: np.full(n, np.nan) for C in C_grid}
    for va, columns in results:
        for C, p in columns.items():
            oof[C][va] = p

    covered = ~np.isnan(oof[C_grid[0]])
    auc = {C: roc_auc_score(y[covered], oof[C][covered]) for C in C_grid}
    best_C = max(auc, key=auc.get)
    return {"C": float(best_C), "inner_oof": oof[best_C], "covered": covered,
            "inner_auc_by_C": {float(C): float(v) for C, v in auc.items()},
            "inner_auc": float(auc[best_C])}


def greedy_select(oof_matrix, y, names, policy="max_f1", min_recall=0.5, covered=None):
    """Greedy forward selection of a soft-voting subset, using only inner out-of-fold scores.

    Soft voting is the mean of the member probability columns, so each candidate subset is
    evaluated by averaging columns of ``oof_matrix``. No model is refit. The threshold is
    re-chosen for every candidate subset, also on the inner out-of-fold scores, so the
    selection objective and the deployed decision rule are the same quantity.

    The returned subset is the best step of the path, exactly the EX_024 rule, but now the
    whole search happens inside one outer fold and is therefore part of the procedure being
    evaluated rather than a decision made against the test data.
    """
    if covered is None:
        covered = np.ones(len(y), dtype=bool)
    y_in = np.asarray(y)[covered]
    columns = {name: oof_matrix[:, j][covered] for j, name in enumerate(names)}

    remaining, selected, path = list(names), [], []
    while remaining:
        scored = []
        for candidate in remaining:
            members = selected + [candidate]
            mean_proba = np.mean([columns[m] for m in members], axis=0)
            threshold = choose_threshold(y_in, mean_proba, policy=policy, min_recall=min_recall)
            objective = f1_score(y_in, (mean_proba >= threshold).astype(int), zero_division=0)
            scored.append((objective, threshold, candidate))
        objective, threshold, candidate = max(scored, key=lambda row: (row[0], -len(row[2])))
        selected.append(candidate)
        remaining.remove(candidate)
        path.append({"n_members": len(selected), "added": candidate,
                     "members": tuple(selected), "objective": float(objective),
                     "threshold": float(threshold)})

    best = max(path, key=lambda row: row["objective"])
    return {"subset": list(best["members"]), "threshold": best["threshold"],
            "objective": best["objective"], "path": pd.DataFrame(path)}


@dataclass
class ProcedureResult:
    """Everything one protocol produces: per-fold records plus assembled out-of-fold scores."""
    protocol: str
    policy: str
    folds: list = field(default_factory=list)
    member_proba: dict = field(default_factory=dict)   # (seed, representation) -> OOF column
    ensemble_proba: dict = field(default_factory=dict)  # seed -> OOF column of the procedure
    ensemble_threshold: dict = field(default_factory=dict)  # (seed, fold) -> frozen threshold


def run_nested_procedure(X, y, folds, names, binary_names, C_grid, checkpoints,
                         protocol="scaffold", policy="max_f1", min_recall=0.5,
                         inner_splits=5, groups=None, n_jobs=1, verbose=True):
    """Runs the full procedure once per outer fold: tune C, build the inner OOF matrix,
    select the subset, fix the threshold, then score the untouched outer test fold.

    What is reported afterwards is no longer "the ensemble of 7". It is the procedure
    "greedy forward selection + soft voting", evaluated as many times as there are outer
    folds, with the selection frequency of each representation as an extra, more informative
    result than any single subset.
    """
    y = np.asarray(y)
    n = len(y)
    result = ProcedureResult(protocol=protocol, policy=policy)
    seeds = sorted({seed for seed, _, _, _ in folds})
    for seed in seeds:
        result.ensemble_proba[seed] = np.full(n, np.nan)
        for name in names:
            result.member_proba[(seed, name)] = np.full(n, np.nan)

    for seed, k, tr, te in folds:
        # The member fits depend on the protocol and the fold, never on the threshold policy,
        # so the policy stays out of their checkpoint key and every policy reuses the same
        # fits. Only the selection, which reads the stored out-of-fold matrix, is per policy.
        fold_tag = f"{protocol}_seed{seed}_fold{k}"
        tag = f"{protocol}_{policy}_seed{seed}_fold{k}"

        if groups is None:
            inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
            inner_folds = list(inner_cv.split(np.zeros(len(tr)), y[tr]))
        else:
            inner_cv = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True,
                                            random_state=seed)
            inner_folds = list(inner_cv.split(np.zeros(len(tr)), y[tr],
                                              groups=np.asarray(groups)[tr]))

        oof_columns, test_columns, chosen_C = {}, {}, {}
        for name in names:
            def fit_one(name=name):
                binary = name in binary_names
                inner = inner_oof_and_C(X[name][tr], y[tr], inner_folds, C_grid,
                                        binary=binary, n_jobs=n_jobs)
                model = build_pipeline(binary=binary, C=inner["C"]).fit(X[name][tr], y[tr])
                return {"C": inner["C"], "inner_oof": inner["inner_oof"],
                        "covered": inner["covered"], "inner_auc": inner["inner_auc"],
                        "test_proba": model.predict_proba(X[name][te])[:, 1]}

            record = checkpoints.run_or_load(f"member_{fold_tag}_{name}", fit_one, verbose=False)
            oof_columns[name] = record["inner_oof"]
            test_columns[name] = record["test_proba"]
            chosen_C[name] = record["C"]
            result.member_proba[(seed, name)][te] = record["test_proba"]

        oof_matrix = np.column_stack([oof_columns[name] for name in names])
        covered = ~np.isnan(oof_matrix).any(axis=1)
        selection = checkpoints.run_or_load(
            f"selection_{tag}",
            lambda: greedy_select(oof_matrix, y[tr], names, policy=policy,
                                  min_recall=min_recall, covered=covered),
            verbose=False)

        subset, threshold = selection["subset"], selection["threshold"]
        ensemble_test = np.mean([test_columns[name] for name in subset], axis=0)
        result.ensemble_proba[seed][te] = ensemble_test
        result.ensemble_threshold[(seed, k)] = threshold

        fold_metrics = all_metrics(y[te], ensemble_test, threshold)
        result.folds.append({
            "protocol": protocol, "policy": policy, "seed": seed, "fold": k,
            "n_train": len(tr), "n_test": len(te), "subset": tuple(subset),
            "n_members": len(subset), "threshold": threshold,
            "inner_objective": selection["objective"],
            "C_by_representation": dict(chosen_C), **fold_metrics,
        })
        if verbose:
            print(f"[{protocol}/{policy}] seed {seed} fold {k}: "
                  f"{len(subset)} members, threshold {threshold:.2f}, "
                  f"test F1 {fold_metrics['f1']:.4f}, ROC-AUC {fold_metrics['roc_auc']:.4f}")

    return result


def selection_stability(result, names):
    """How often each representation survives the in-fold greedy selection.

    A representation chosen in 14 of 15 folds is a robust finding. A subset that changes
    every fold says the single subset reported by EX_024 was a property of those partitions,
    and saying so is a stronger result than hiding it.
    """
    n_folds = len(result.folds)
    counts = {name: 0 for name in names}
    for record in result.folds:
        for name in record["subset"]:
            counts[name] += 1
    rows = [{"representation": name, "times_selected": counts[name], "n_folds": n_folds,
             "selection_frequency": counts[name] / n_folds} for name in names]
    return (pd.DataFrame(rows).sort_values("times_selected", ascending=False)
            .reset_index(drop=True))


def oof_metrics_by_seed(result, y, groups, names, n_boot=0, seed=RANDOM_STATE):
    """Metrics on the assembled out-of-fold predictions, one complete pass per repetition.

    Every repetition predicts each compound exactly once, so these are complete samples and
    the right object to bootstrap. The threshold used is the mean of the thresholds the folds
    of that repetition froze.
    """
    y = np.asarray(y)
    rows = []
    for repetition, proba in result.ensemble_proba.items():
        thresholds = [t for (s, _), t in result.ensemble_threshold.items() if s == repetition]
        threshold = float(np.mean(thresholds))
        covered = ~np.isnan(proba)
        rows.append({"model": "procedure (greedy + soft voting)", "seed": repetition,
                     "threshold": threshold, "n_scored": int(covered.sum()),
                     **all_metrics(y[covered], proba[covered], threshold)})
        for name in names:
            member = result.member_proba[(repetition, name)]
            covered_member = ~np.isnan(member)
            rows.append({"model": name, "seed": repetition, "threshold": threshold,
                         "n_scored": int(covered_member.sum()),
                         **all_metrics(y[covered_member], member[covered_member], threshold)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 7. The lock-box
# --------------------------------------------------------------------------------------


@dataclass
class FrozenConfig:
    """The configuration written to disk before the external test set is read.

    The value of a held-out set is procedural, not statistical. Cross-validation is an estimate
    that has been looked at many times: every protocol decision consulted it, and consulting is
    a mild form of fitting. A set read exactly once, after everything is fixed, is the only
    measurement that cannot be accused of having been optimized.

    Running the external set, seeing a weak number and changing anything above destroys that
    property. From then on it is a validation set, not a test set, and must be described as one.
    """
    members: list
    C_by_member: dict
    weights: dict
    threshold: float
    policy: str
    protocol: str
    selection_rule: str
    created_at: str
    notes: str = ""

    def fingerprint(self):
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        import hashlib
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def freeze_config(path, **kwargs):
    """Writes the frozen configuration, refusing to overwrite an existing one.

    Refusing the overwrite is the whole point: once a configuration is frozen and the external
    set has been read, silently replacing the file would erase the evidence of what was
    actually locked. Delete the file by hand, and say so in the manuscript, if a protocol
    really has to change.
    """
    config = FrozenConfig(created_at=pd.Timestamp.now().isoformat(), **kwargs)
    if os.path.exists(path):
        existing = json.load(open(path))
        print(f"[lock-box] {path} already exists (fingerprint {existing.get('fingerprint')}); "
              f"keeping it. Delete the file by hand to re-freeze.")
        return existing
    payload = {**asdict(config), "fingerprint": config.fingerprint()}
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print(f"[lock-box] frozen {path} (fingerprint {payload['fingerprint']})")
    return payload


def log_external_read(path, config_fingerprint, note=""):
    """Appends one record per external-set read, so the count is auditable after the fact.

    A second entry with a different fingerprint is the signal that the set was consulted,
    the protocol was then changed, and the number must no longer be called a held-out result.
    """
    log = json.load(open(path)) if os.path.exists(path) else []
    log.append({"read_at": pd.Timestamp.now().isoformat(),
                "config_fingerprint": config_fingerprint, "note": note})
    with open(path, "w") as handle:
        json.dump(log, handle, indent=2)
    fingerprints = {entry["config_fingerprint"] for entry in log}
    if len(log) > 1:
        print(f"[lock-box] WARNING: the external set has now been read {len(log)} times.")
        if len(fingerprints) > 1:
            print("[lock-box] WARNING: reads span more than one configuration. From the second "
                  "configuration onward this is a validation set, not a held-out test set.")
    return log


# --------------------------------------------------------------------------------------
# 8. Label-conflict ablation
# --------------------------------------------------------------------------------------


def conflict_candidates(dataframe, source_column="sources", label_column="label"):
    """Rows whose positive label depends on the "any source says positive wins" rule.

    The merged dataset keeps one row per compound with the union of its sources and the
    winning label, so the per-source labels that produced a conflict are no longer recoverable
    from it. What is recoverable, and auditable, is the set of positives that come from more
    than one source: every genuine conflict is in this set, so it is a superset of them. The
    ablation over this superset is therefore a conservative version of the intended test.

    If the raw per-source tables are restored, pass an explicit index list instead and the
    exact conflicts can be used.
    """
    sources = dataframe[source_column].fillna("").apply(
        lambda s: [x.strip() for x in s.split(";") if x.strip()])
    multi_source = sources.apply(len) > 1
    return np.flatnonzero((dataframe[label_column].to_numpy() == 1) & multi_source.to_numpy())


def apply_conflict_policy(y, conflict_index, policy):
    """``positive`` keeps the current rule, ``negative`` flips them, ``removed`` drops them.

    Returns the labels and a row mask, so the caller slices the embeddings the same way.
    """
    y = np.asarray(y).copy()
    keep = np.ones(len(y), dtype=bool)
    if policy == "positive":
        pass
    elif policy == "negative":
        y[conflict_index] = 0
    elif policy == "removed":
        keep[conflict_index] = False
    else:
        raise ValueError(f"unknown policy: {policy}")
    return y, keep


def ranking_under_policy(X, y_policy, keep, names, binary_names, groups, C_grid, checkpoints,
                         policy_name, n_splits=5, seed=RANDOM_STATE, n_jobs=1):
    """Ranks every representation under one labelling policy, on scaffold-grouped folds.

    The three policies are compared only through the resulting order of the representations,
    so all that matters is that they are treated identically: the same folds, the same grid
    and the same C-selection rule. C is selected on the same grouped folds that produce the
    out-of-fold scores, which is mildly optimistic in absolute terms and identical across the
    three policies, which is what the comparison needs.
    """
    y_policy = np.asarray(y_policy)[keep]
    groups_policy = np.asarray(groups)[keep]
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(np.zeros(len(y_policy)), y_policy, groups=groups_policy))

    rows = {}
    for name in names:
        def fit_one(name=name):
            return inner_oof_and_C(X[name][keep], y_policy, folds, C_grid,
                                   binary=name in binary_names, n_jobs=n_jobs)

        record = checkpoints.run_or_load(f"ablation_{policy_name}_{name}", fit_one, verbose=False)
        covered = record["covered"]
        threshold = choose_threshold(y_policy[covered], record["inner_oof"][covered], "max_f1")
        rows[name] = {"C": record["C"],
                      **all_metrics(y_policy[covered], record["inner_oof"][covered], threshold)}
    ranking = pd.DataFrame(rows).T
    ranking.index.name = "representation"
    return ranking.sort_values("roc_auc", ascending=False)


def fit_frozen_members(X, y, members, C_by_member, binary_names, checkpoints, tag="frozen"):
    """Fits each frozen member on the whole primary dataset with its frozen C.

    This happens before the external set is read. No hyperparameter, no member and no
    threshold is revisited afterwards.
    """
    fitted = {}
    for name in members:
        def fit_one(name=name):
            return build_pipeline(binary=name in binary_names,
                                  C=float(C_by_member[name])).fit(X[name], np.asarray(y))

        fitted[name] = checkpoints.run_or_load(f"{tag}_member_{name}", fit_one)
    return fitted


def soft_vote(fitted, X_eval, members, weights=None):
    """Soft voting: the (optionally weighted) mean of the member probability columns."""
    columns = np.column_stack([fitted[name].predict_proba(X_eval[name])[:, 1]
                               for name in members])
    w = None if weights is None else np.asarray([weights[name] for name in members], dtype=float)
    return np.average(columns, axis=1, weights=w)


def external_cuts(chem_primary, chem_external, level="L5_inchikey"):
    """The two cuts of the external set that matter, as boolean masks over its rows.

    ``full``            everything except compounds that are the same standardized substance
                        as a primary-dataset compound. Canonical-SMILES deduplication does not
                        catch a salt, a charge state or a tautomer of a training compound, so
                        the overlap is removed at the standardized level instead.
    ``novel_scaffold``  the subset of ``full`` whose Bemis-Murcko scaffold never appears in the
                        primary dataset. This is the discovery number: a chemotype the model
                        has never seen in any form.
    """
    primary_ids = set(chem_primary[level].dropna())
    primary_scaffolds = set(chem_primary["group"].dropna())

    same_substance = chem_external[level].isin(primary_ids).to_numpy()
    full = ~same_substance
    novel_scaffold = full & (~chem_external["group"].isin(primary_scaffolds).to_numpy())
    return {"full": full, "novel_scaffold": novel_scaffold,
            "n_same_substance_removed": int(same_substance.sum())}


def ranking_correlation(ranking_a, ranking_b, metric="roc_auc", n_boot=2000,
                        seed=RANDOM_STATE, alpha=0.05):
    """Spearman correlation between two representation rankings, with a bootstrap interval.

    The ranking of the representations is the central contribution, so what matters is not
    whether the absolute numbers move under a labelling policy but whether the order does.
    """
    shared = sorted(set(ranking_a.index) & set(ranking_b.index))
    a = ranking_a.loc[shared, metric].to_numpy(dtype=float)
    b = ranking_b.loc[shared, metric].to_numpy(dtype=float)
    rho, p_value = stats.spearmanr(a, b)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.choice(len(shared), size=len(shared), replace=True)
        if len(np.unique(a[idx])) < 3:
            continue
        draws.append(stats.spearmanr(a[idx], b[idx]).statistic)
    draws = np.asarray([d for d in draws if np.isfinite(d)])
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2]) if draws.size else (np.nan, np.nan)
    return {"metric": metric, "n_representations": len(shared), "spearman_rho": float(rho),
            "p_value": float(p_value), "ci_low": float(low), "ci_high": float(high)}


# --------------------------------------------------------------------------------------
# 9. EX_027 additions
# --------------------------------------------------------------------------------------
#
# Everything in this section is derived from the per-fold records the nested protocol already
# wrote to disk: for each outer fold and each representation, the chosen C, the inner
# out-of-fold probability column, and the probability column of the untouched outer test
# fold. Soft voting is an average of probability columns and a threshold is a cut on one, so
# ensembles, thresholds, selection paths and diversity all follow from those stored columns
# without refitting a single model. The only new fits in EX_027 are the learning curves,
# which by definition need models trained on smaller samples.


# ---- 9.1 Re-reading the nested protocol -------------------------------------------------


@dataclass
class MemberRuns:
    """The stored per-fold output of one protocol, indexed for reuse.

    ``test_proba`` is the probability column of an outer test fold: the only quantity that may
    be used for evaluation. ``inner_oof`` is the out-of-fold column of the inner loop, computed
    from the outer training part alone: the only quantity that may be used for decisions
    (choosing C, choosing members, choosing a threshold).
    """
    protocol: str
    names: list
    folds: list
    test_proba: dict = field(default_factory=dict)      # (seed, fold, name) -> outer test column
    inner_oof: dict = field(default_factory=dict)       # (seed, fold, name) -> inner OOF column
    inner_covered: dict = field(default_factory=dict)   # (seed, fold, name) -> bool mask
    C: dict = field(default_factory=dict)               # (seed, fold, name) -> chosen C
    inner_auc: dict = field(default_factory=dict)       # (seed, fold, name) -> inner ROC-AUC

    def train_index(self):
        return {(seed, k): tr for seed, k, tr, _ in self.folds}

    def test_index(self):
        return {(seed, k): te for seed, k, _, te in self.folds}

    def seeds(self):
        return sorted({seed for seed, _, _, _ in self.folds})


def load_member_runs(checkpoints, protocol, folds, names):
    """Loads every ``member_*`` checkpoint of one protocol. Raises if one is missing.

    A missing checkpoint means the nested protocol has not been run for that fold, and the
    right answer is to run it rather than to quietly report a table with holes in it.
    """
    runs = MemberRuns(protocol=protocol, names=list(names), folds=list(folds))
    missing = []
    for seed, k, _, _ in folds:
        for name in names:
            key = f"member_{protocol}_seed{seed}_fold{k}_{name}"
            if not checkpoints.has(key):
                missing.append(key)
                continue
            record = checkpoints.load(key)
            runs.test_proba[(seed, k, name)] = record["test_proba"]
            runs.inner_oof[(seed, k, name)] = record["inner_oof"]
            runs.inner_covered[(seed, k, name)] = record["covered"]
            runs.C[(seed, k, name)] = float(record["C"])
            runs.inner_auc[(seed, k, name)] = float(record["inner_auc"])
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} member checkpoints are missing for protocol '{protocol}'. "
            f"First: {missing[0]}. Run the nested procedure for this protocol first.")
    return runs


def member_thresholds(runs, y, policy="max_f1", min_recall=0.5):
    """One threshold per (fold, representation), chosen on that fold's inner out-of-fold scores.

    This is the correction EX_024 needs most for its central table. 0.5 is the point where
    P(positive) > P(negative) for a calibrated, balanced classifier; this dataset is 17.2%
    positive and uses ``class_weight="balanced"``, which shifts the probabilities without
    calibrating them, so 0.5 is neither the optimum nor a neutral point. Reading twenty
    representations at one shared, arbitrary cut can even invert their order, because their
    score distributions differ. Each representation now gets its own cut, chosen inside the
    fold and never on the test data.
    """
    y = np.asarray(y)
    train = runs.train_index()
    out = {}
    for (seed, k, name), inner in runs.inner_oof.items():
        covered = runs.inner_covered[(seed, k, name)]
        y_inner = y[train[(seed, k)]][covered]
        out[(seed, k, name)] = choose_threshold(y_inner, inner[covered], policy=policy,
                                                min_recall=min_recall)
    return out


def member_fold_table(runs, y, policy="max_f1", min_recall=0.5):
    """Per-fold metrics of every individual representation, each at its own in-fold threshold.

    This is the EX_024 section 4.x table, recomputed so that the threshold-dependent columns
    (F1, precision, recall, MCC, balanced accuracy) are no longer read at the sklearn default,
    and with the ranking metrics (Precision@k, EF@k, BEDROC) added, because virtual screening
    ranks a library instead of classifying it.
    """
    y = np.asarray(y)
    thresholds = member_thresholds(runs, y, policy=policy, min_recall=min_recall)
    rows = []
    for seed, k, tr, te in runs.folds:
        for name in runs.names:
            threshold = thresholds[(seed, k, name)]
            proba = runs.test_proba[(seed, k, name)]
            rows.append({
                "protocol": runs.protocol, "policy": policy, "model": name,
                "group": "individual", "seed": seed, "fold": k,
                "n_train": len(tr), "n_test": len(te),
                "C": runs.C[(seed, k, name)], "inner_auc": runs.inner_auc[(seed, k, name)],
                **all_metrics(y[te], proba, threshold),
            })
    return pd.DataFrame(rows)


def assemble_oof(runs, y, policy="max_f1", min_recall=0.5):
    """Assembles, per repetition, the out-of-fold score vector of every representation.

    Each repetition predicts every compound exactly once, so these vectors are complete
    samples of the dataset and are the right object to bootstrap. The threshold attached to a
    repetition is the mean of the thresholds its folds froze, which keeps the decision rule a
    product of training data only.
    """
    y = np.asarray(y)
    n = len(y)
    thresholds = member_thresholds(runs, y, policy=policy, min_recall=min_recall)
    proba = {(seed, name): np.full(n, np.nan)
             for seed in runs.seeds() for name in runs.names}
    collected = {(seed, name): [] for seed in runs.seeds() for name in runs.names}
    for seed, k, _, te in runs.folds:
        for name in runs.names:
            proba[(seed, name)][te] = runs.test_proba[(seed, k, name)]
            collected[(seed, name)].append(thresholds[(seed, k, name)])
    threshold = {key: float(np.mean(values)) for key, values in collected.items()}
    return proba, threshold


# ---- 9.2 Voting over all representations -------------------------------------------------

VOTING_VARIANTS = ("soft", "weighted_soft", "hard")


def _inner_matrix(runs, seed, k, names):
    columns = [runs.inner_oof[(seed, k, name)] for name in names]
    matrix = np.column_stack(columns)
    covered = np.column_stack([runs.inner_covered[(seed, k, name)] for name in names]).all(axis=1)
    return matrix, covered


def voting_fold_table(runs, y, policy="max_f1", min_recall=0.5, variants=VOTING_VARIANTS,
                      subset=None, label_prefix=""):
    """The EX_024 section 6 ensembles, rebuilt with every decision taken inside the fold.

    Three variants, all evaluated on the untouched outer fold:

    ``soft``           the unweighted mean of the member probability columns.
    ``weighted_soft``  the mean weighted by each member's INNER out-of-fold ROC-AUC. EX_024
                       weighted by the outer test ROC-AUC, which lets the test fold choose the
                       weights; the inner estimate is the same idea without the leak.
    ``hard``           the fraction of members voting positive, each at its own in-fold
                       threshold. Its score takes few distinct values, so its ranking metrics
                       carry heavy ties and should be read with that in mind.

    The ensemble threshold is chosen on the inner out-of-fold scores of the same combination
    rule, so the quantity optimized and the quantity deployed are the same.
    """
    y = np.asarray(y)
    names = list(subset) if subset else list(runs.names)
    member_t = member_thresholds(runs, y, policy=policy, min_recall=min_recall)
    train = runs.train_index()
    rows, proba = [], {}
    for variant in variants:
        for seed in runs.seeds():
            proba[(variant, seed)] = np.full(len(y), np.nan)

    for seed, k, tr, te in runs.folds:
        matrix, covered = _inner_matrix(runs, seed, k, names)
        y_inner = y[tr]
        test_matrix = np.column_stack([runs.test_proba[(seed, k, name)] for name in names])
        weights = np.array([max(runs.inner_auc[(seed, k, name)], 0.5) for name in names])
        weights = weights / weights.sum()

        for variant in variants:
            if variant == "soft":
                inner_score = matrix.mean(axis=1)
                test_score = test_matrix.mean(axis=1)
            elif variant == "weighted_soft":
                inner_score = matrix @ weights
                test_score = test_matrix @ weights
            elif variant == "hard":
                cuts = np.array([member_t[(seed, k, name)] for name in names])
                inner_score = (matrix >= cuts).mean(axis=1)
                test_score = (test_matrix >= cuts).mean(axis=1)
            else:
                raise ValueError(f"unknown voting variant: {variant}")

            threshold = choose_threshold(y_inner[covered], inner_score[covered],
                                         policy=policy, min_recall=min_recall)
            label = f"{label_prefix}{variant} voting"
            rows.append({
                "protocol": runs.protocol, "policy": policy, "model": label,
                "group": "ensemble", "seed": seed, "fold": k,
                "n_train": len(tr), "n_test": len(te), "n_members": len(names),
                "C": np.nan, "inner_auc": np.nan,
                **all_metrics(y[te], test_score, threshold),
            })
            proba[(variant, seed)][te] = test_score

    thresholds = {}
    table = pd.DataFrame(rows)
    for variant in variants:
        for seed in runs.seeds():
            selected = table[(table["model"] == f"{label_prefix}{variant} voting")
                             & (table["seed"] == seed)]
            thresholds[(variant, seed)] = float(selected["threshold"].mean())
    return table, proba, thresholds


# ---- 9.3 What selecting against the evaluation data is worth ------------------------------


def oracle_selection_fold_table(runs, y, policy="max_f1", min_recall=0.5):
    """The same greedy search, but allowed to read the outer test fold. An upper bound.

    EX_024 chose its subset of representations by maximizing F1 on cross-validation over the
    same 4,233 compounds it then reported cross-validation on. 2^20 - 1 subsets exist and the
    greedy path visits 210 of them; even 210 attempts against the same scores buy visible
    optimism. Running the identical search directly on the evaluation fold measures how large
    that effect can be in these folds, which turns "the selection is biased" into a number.
    """
    y = np.asarray(y)
    rows = []
    for seed, k, tr, te in runs.folds:
        test_matrix = np.column_stack([runs.test_proba[(seed, k, name)] for name in runs.names])
        selection = greedy_select(test_matrix, y[te], runs.names, policy=policy,
                                  min_recall=min_recall)
        rows.append({
            "protocol": runs.protocol, "policy": policy,
            "model": "oracle selection (upper bound, not a result)",
            "group": "contrast", "seed": seed, "fold": k,
            "n_train": len(tr), "n_test": len(te),
            "subset": tuple(selection["subset"]), "n_members": len(selection["subset"]),
            **all_metrics(y[te], np.mean([test_matrix[:, runs.names.index(nm)]
                                          for nm in selection["subset"]], axis=0),
                          selection["threshold"]),
        })
    return pd.DataFrame(rows)


# ---- 9.4 The greedy path, fold by fold ---------------------------------------------------


def greedy_path_table(checkpoints, protocol, policy, folds):
    """Concatenates the greedy path of every outer fold.

    EX_024 plots one path. Fifteen paths say something EX_024 could not: whether the shape of
    the curve, and the subset size it peaks at, are properties of the data or of a partition.
    """
    rows = []
    for seed, k, _, _ in folds:
        key = f"selection_{protocol}_{policy}_seed{seed}_fold{k}"
        if not checkpoints.has(key):
            raise FileNotFoundError(f"missing selection checkpoint {key}")
        path = checkpoints.load(key)["path"].copy()
        path["seed"], path["fold"], path["protocol"], path["policy"] = seed, k, protocol, policy
        rows.append(path)
    return pd.concat(rows, ignore_index=True)


def greedy_path_summary(path_table, alpha=0.05):
    """Median and percentile spread of the inner objective at each subset size, over folds.

    The spread here is descriptive: it is the variability of the greedy path across
    partitions, not a confidence interval for a population quantity.
    """
    grouped = path_table.groupby("n_members")["objective"]
    return pd.DataFrame({
        "n_members": sorted(path_table["n_members"].unique()),
        "median": grouped.median().to_numpy(),
        "low": grouped.quantile(alpha / 2).to_numpy(),
        "high": grouped.quantile(1 - alpha / 2).to_numpy(),
        "n_folds": grouped.size().to_numpy(),
    })


def subset_size_stability(fold_records):
    """How many members the in-fold selection keeps, fold by fold."""
    sizes = fold_records["n_members"].to_numpy()
    return {"median_size": float(np.median(sizes)), "min_size": int(sizes.min()),
            "max_size": int(sizes.max()), "n_distinct_subsets":
            int(len(set(fold_records["subset"]))), "n_folds": int(len(sizes))}


# ---- 9.5 Learning curves ------------------------------------------------------------------

LEARNING_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)


def _subsample(index, y, fraction, groups=None, seed=RANDOM_STATE):
    """A training subsample of the requested size, drawn by group when groups are given.

    Under the scaffold protocol the unit of independence is the scaffold, so a learning curve
    that subsamples compounds lets members of a congeneric series re-enter the smaller
    training sets and measures memorization recovering, not learning. Drawing whole groups
    keeps the curve consistent with the split it is supposed to describe.
    """
    index = np.asarray(index)
    rng = np.random.default_rng(seed)
    target = max(int(round(fraction * len(index))), 20)
    if fraction >= 1.0:
        return index
    if groups is None:
        y_local = np.asarray(y)[index]
        order = rng.permutation(len(index))
        taken, counts = [], {0: 0, 1: 0}
        quota = {label: max(int(round(fraction * (y_local == label).sum())), 1)
                 for label in (0, 1)}
        for position in order:
            label = int(y_local[position])
            if counts[label] < quota[label]:
                taken.append(index[position])
                counts[label] += 1
        return np.asarray(sorted(taken))
    local_groups = np.asarray(groups)[index]
    unique = rng.permutation(np.unique(local_groups))
    chosen, total = [], 0
    for group in unique:
        members = index[local_groups == group]
        chosen.append(members)
        total += len(members)
        if total >= target:
            break
    return np.sort(np.concatenate(chosen))


def learning_curve_grouped(X_rep, y, folds, C_by_fold, binary=False, groups=None,
                           fractions=LEARNING_FRACTIONS, n_jobs=1, seed=RANDOM_STATE):
    """Train and test ROC-AUC and PR-AUC as the training sample grows.

    Both reported metrics are threshold-free on purpose. A learning curve of F1 would need a
    threshold at every sample size; choosing it on the subsample's own fitted scores makes the
    curve optimistic, and carrying over a threshold chosen at full size makes the curve a
    picture of threshold drift. ROC-AUC and PR-AUC answer the question the curve is asked
    -- is there more to gain from more data -- without that confound.

    C is held at the value the fold's inner loop selected, so the curve isolates sample size.
    """
    from joblib import Parallel, delayed

    y = np.asarray(y)

    def one(seed_id, k, tr, te, fraction):
        subsample = _subsample(tr, y, fraction, groups=groups, seed=seed + 1000 * k + int(
            round(100 * fraction)))
        if len(np.unique(y[subsample])) < 2:
            return None
        model = build_pipeline(binary=binary, C=float(C_by_fold[(seed_id, k)]))
        model.fit(X_rep[subsample], y[subsample])
        train_score = model.predict_proba(X_rep[subsample])[:, 1]
        test_score = model.predict_proba(X_rep[te])[:, 1]
        n_groups = (len(np.unique(np.asarray(groups)[subsample])) if groups is not None
                    else len(subsample))
        return {
            "seed": seed_id, "fold": k, "fraction": fraction,
            "n_train_used": int(len(subsample)), "n_train_groups_used": int(n_groups),
            "train_roc_auc": float(roc_auc_score(y[subsample], train_score)),
            "test_roc_auc": float(roc_auc_score(y[te], test_score)),
            "train_average_precision": float(average_precision_score(y[subsample], train_score)),
            "test_average_precision": float(average_precision_score(y[te], test_score)),
        }

    jobs = [delayed(one)(seed_id, k, tr, te, fraction)
            for seed_id, k, tr, te in folds for fraction in fractions]
    rows = [row for row in Parallel(n_jobs=n_jobs)(jobs) if row is not None]
    return pd.DataFrame(rows)


def learning_curve_summary(table, alpha=0.05):
    """Median curve with a percentile band over folds, for plotting."""
    metrics = ["train_roc_auc", "test_roc_auc",
               "train_average_precision", "test_average_precision"]
    grouped = table.groupby("fraction")
    out = {"fraction": sorted(table["fraction"].unique()),
           "n_train_used": grouped["n_train_used"].median().to_numpy()}
    for metric in metrics:
        out[f"{metric}_median"] = grouped[metric].median().to_numpy()
        out[f"{metric}_low"] = grouped[metric].quantile(alpha / 2).to_numpy()
        out[f"{metric}_high"] = grouped[metric].quantile(1 - alpha / 2).to_numpy()
    return pd.DataFrame(out)


def plot_learning_curve(summary, title="", metric="roc_auc", ax=None):
    """Plots one summarized learning curve: train and test, with the fold band."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.4))
    x = summary["n_train_used"].to_numpy()
    for part, color in (("train", "tab:blue"), ("test", "tab:red")):
        key = f"{part}_{metric}"
        ax.plot(x, summary[f"{key}_median"], marker="o", color=color, label=part)
        ax.fill_between(x, summary[f"{key}_low"], summary[f"{key}_high"],
                        color=color, alpha=0.15)
    ax.set_xlabel("training compounds used")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


# ---- 9.6 Diversity ------------------------------------------------------------------------


def _counts(correct_a, correct_b):
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    return {"n11": int((a & b).sum()), "n10": int((a & ~b).sum()),
            "n01": int((~a & b).sum()), "n00": int((~a & ~b).sum()), "n": int(a.size)}


def disagreement_measure(correct_a, correct_b):
    """Fraction of compounds the two models classify differently. Higher means more diverse."""
    c = _counts(correct_a, correct_b)
    return (c["n10"] + c["n01"]) / c["n"]


def double_fault_measure(correct_a, correct_b):
    """Fraction of compounds both models get wrong. Lower means more complementary."""
    c = _counts(correct_a, correct_b)
    return c["n00"] / c["n"]


def q_statistic(correct_a, correct_b):
    """Yule's Q on the correctness pair. 0 is independence, 1 is fully coincident errors."""
    c = _counts(correct_a, correct_b)
    numerator = c["n11"] * c["n00"] - c["n01"] * c["n10"]
    denominator = c["n11"] * c["n00"] + c["n01"] * c["n10"]
    return numerator / denominator if denominator else np.nan


def correlation_measure(correct_a, correct_b):
    """Correlation coefficient between the two correctness vectors."""
    c = _counts(correct_a, correct_b)
    denominator = np.sqrt(float((c["n11"] + c["n10"]) * (c["n01"] + c["n00"])
                                * (c["n11"] + c["n01"]) * (c["n10"] + c["n00"])))
    if denominator == 0:
        return np.nan
    return (c["n11"] * c["n00"] - c["n01"] * c["n10"]) / denominator


def pairwise_matrix(function, correct, names):
    """Square matrix of a pairwise diversity measure, with the diagonal left at NaN."""
    size = len(names)
    matrix = np.full((size, size), np.nan)
    for i in range(size):
        for j in range(size):
            if i != j:
                matrix[i, j] = function(correct[names[i]], correct[names[j]])
    return matrix


def correctness_from_oof(proba, threshold, y, names, seed):
    """Out-of-fold correctness vectors of a set of models, each at its own threshold.

    Diversity is a property of where models make mistakes, so the mistakes have to be the ones
    the deployed rule would make: each member is cut at its own in-fold threshold, not at a
    shared default.
    """
    y = np.asarray(y)
    covered = np.ones(len(y), dtype=bool)
    for name in names:
        covered &= ~np.isnan(proba[(seed, name)])
    correct, predictions = {}, {}
    for name in names:
        prediction = (proba[(seed, name)][covered] >= threshold[(seed, name)]).astype(int)
        predictions[name] = prediction
        correct[name] = prediction == y[covered]
    return correct, predictions, covered


def diversity_summary(correct, names):
    """Per-member accuracy next to its mean disagreement and double-fault with the others."""
    disagreement = pairwise_matrix(disagreement_measure, correct, names)
    double_fault = pairwise_matrix(double_fault_measure, correct, names)
    return pd.DataFrame({
        "member": names,
        "oof_accuracy": [float(np.mean(correct[name])) for name in names],
        "mean_disagreement": np.nanmean(disagreement, axis=1),
        "mean_double_fault": np.nanmean(double_fault, axis=1),
    }).sort_values("oof_accuracy", ascending=False).reset_index(drop=True)


def oracle_and_exclusive(correct, names):
    """Oracle accuracy, the full-failure rate, and each member's exclusive contribution.

    The oracle is the ceiling any combination rule could reach if it always trusted the right
    member. The gap between it and the ensemble is what a better combination rule could still
    buy. A member whose exclusive count is zero adds nothing the others do not already cover.
    """
    matrix = np.column_stack([correct[name] for name in names])
    n_correct = matrix.sum(axis=1)
    exclusive = {name: int(((matrix[:, j]) & (n_correct == 1)).sum())
                 for j, name in enumerate(names)}
    return {
        "n_scored": int(matrix.shape[0]),
        "oracle_accuracy": float((n_correct >= 1).mean()),
        "all_wrong_rate": float((n_correct == 0).mean()),
        "all_right_rate": float((n_correct == len(names)).mean()),
        "majority_accuracy": float((n_correct > len(names) / 2).mean()),
        "exclusive_errors_fixed": exclusive,
    }


# ---- 9.7 Reporting: intervals instead of mean +/- SD --------------------------------------


def fold_descriptive(table, metrics, by=("protocol", "policy", "model")):
    """Median and interquartile range of a metric across folds, explicitly descriptive.

    ``mean +/- SD`` over 15 folds looks like inference and is not. The spread between folds
    measures the variability of the partitioning, and the 15 folds are not independent: they
    are 3 repetitions over the same compounds, so their training sets overlap heavily. These
    columns are kept because they describe the run, and they are never used to support a
    claim; the bootstrap tables below do that.
    """
    grouped = table.groupby(list(by))
    out = {}
    for metric in metrics:
        out[f"{metric}_median"] = grouped[metric].median()
        out[f"{metric}_q1"] = grouped[metric].quantile(0.25)
        out[f"{metric}_q3"] = grouped[metric].quantile(0.75)
    return pd.DataFrame(out).reset_index()


def oof_ci_table(y, proba, threshold, models, groups, metrics=None, n_boot=2000,
                 seed=RANDOM_STATE, alpha=0.05, by_seed=None, label=None):
    """Group-bootstrap intervals on the assembled out-of-fold predictions of several models.

    The resampling unit is the group the protocol split on -- a whole Bemis-Murcko scaffold
    under the scaffold protocol -- because compounds sharing a skeleton are dependent and
    resampling them individually narrows the interval artificially.
    """
    metrics = metrics or ["roc_auc", "average_precision", "bedroc_20", "f1", "precision",
                          "recall", "matthews_corrcoef", "precision_at_100", "ef_100"]
    y = np.asarray(y)
    rows = []
    for model in models:
        key = (by_seed, model) if by_seed is not None else model
        scores = proba[key]
        covered = ~np.isnan(scores)
        table = group_bootstrap_ci(y[covered], scores[covered], np.asarray(groups)[covered],
                                   threshold[key], metrics=metrics, n_boot=n_boot,
                                   seed=seed, alpha=alpha)
        table.insert(0, "model", label(model) if callable(label) else model)
        table["n_scored"] = int(covered.sum())
        table["threshold"] = threshold[key]
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def format_ci(ci_table, decimals=3, value="point"):
    """Pivots a bootstrap table into 'point [low, high]' strings, one row per model."""
    formatted = ci_table.copy()
    formatted["text"] = [
        f"{row[value]:.{decimals}f} [{row['ci_low']:.{decimals}f}, {row['ci_high']:.{decimals}f}]"
        for _, row in formatted.iterrows()]
    wide = formatted.pivot_table(index="model", columns="metric", values="text",
                                 aggfunc="first")
    wide.columns = [METRIC_LABELS.get(c, c) for c in wide.columns]
    return wide


def paired_against(y, proba, threshold, challengers, reference, groups, metrics=None,
                   n_boot=2000, seed=RANDOM_STATE, alpha=0.05):
    """Paired group-bootstrap of every challenger minus one reference model.

    Two overlapping one-model intervals do not establish that a difference is absent when the
    models are correlated, and models scored on the same compounds always are. The paired
    interval on the difference is the quantity that answers "does A beat B".
    """
    metrics = metrics or ["roc_auc", "average_precision", "bedroc_20", "f1",
                          "precision_at_100"]
    y = np.asarray(y)
    rows = []
    for challenger in challengers:
        shared = (~np.isnan(proba[challenger])) & (~np.isnan(proba[reference]))
        table = paired_group_bootstrap(
            y[shared], proba[challenger][shared], proba[reference][shared],
            np.asarray(groups)[shared], threshold[challenger], threshold[reference],
            metrics=metrics, n_boot=n_boot, seed=seed, alpha=alpha)
        table.insert(0, "challenger", challenger)
        table.insert(1, "reference", reference)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def ranking_table(ci_table, order_by="ROC-AUC"):
    """Final ranking: one row per model, every cell an interval, ordered by one metric."""
    wide = format_ci(ci_table)
    points = ci_table.pivot_table(index="model", columns="metric", values="point",
                                  aggfunc="first")
    points.columns = [METRIC_LABELS.get(c, c) for c in points.columns]
    if order_by in points.columns:
        wide = wide.loc[points[order_by].sort_values(ascending=False).index]
    return wide
