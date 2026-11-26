"""The protocol: nested fits, metrics, ensembles and final models.

Runs the pipeline over a chemically standardized dataset, where structural duplicates are
merged before the embeddings are computed. From there:

* L2 logistic regression per representation, with C chosen by nested inner CV, over 5 folds x
  3 seeds, under both the scaffold and the random split;
* metrics at threshold 0.5, on train and test, per fold and averaged;
* three ensembles over all representations (soft voting, ROC-AUC-weighted soft voting,
  stacking) plus greedy selection, all decided from the training part of each fold alone;
* final models fitted on the whole dataset.

Standardization (RDKit), the folds, the logistic-regression pipeline, the inner CV and the
diversity measures come from core_utils, so that every experiment shares exactly the same code
in those parts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

from core_utils import (RANDOM_STATE, Checkpoints, bedroc, build_pipeline,  # noqa: F401
                         chemistry_table, correlation_measure, disagreement_measure,
                         double_fault_measure, duplicate_audit, external_cuts, inner_oof_and_C,
                         log_external_read, pairwise_matrix, precision_at_k, q_statistic,
                         repeated_random_folds, repeated_scaffold_folds)

DEFAULT_THRESHOLD = 0.5
METRICS = ["accuracy", "recall", "precision", "f1", "balanced_accuracy", "mcc", "roc_auc",
            "average_precision"]
METRIC_NAMES = {"accuracy": "Accuracy", "recall": "Recall", "precision": "Precision",
                  "f1": "F1", "balanced_accuracy": "Balanced accuracy", "mcc": "MCC",
                  "roc_auc": "ROC-AUC", "average_precision": "AP (PR-AUC)"}
SCREENING_K = (50, 100, 200)
ENSEMBLES = ["Soft voting", "Weighted soft voting (ROC-AUC)", "Stacking", "Greedy (F1)"]
THRESHOLD_GRID = np.round(np.arange(0.40, 0.6001, 0.01), 2)


# --------------------------------------------------------------------------------------
# 1. Chemical standardization
# --------------------------------------------------------------------------------------

def standardize_dataset(dataset, smiles_column="smiles", label_column="label"):
    """Standardizes the dataset and merges structural duplicates.

    Every SMILES goes through the same standardization (cleanup, fragment parent, uncharging,
    canonical tautomer). Rows sharing a standardized InChIKey are the same substance and become
    a single row:

    * SMILES: the standardized canonical tautomer of the first occurrence;
    * label: positive if any form is positive -- the same rule the dataset already uses for
      conflicts between sources;
    * the other columns keep the first occurrence; ``n_forms`` says how many rows were merged
      and ``conflicting_label`` marks the substances whose forms disagreed.

    Molecules RDKit cannot parse are dropped. Returns the standardized dataset, the
    standardization table per original row, and a summary with the counts.
    """
    dataset = dataset.reset_index(drop=True)
    chemistry = chemistry_table(dataset[smiles_column].astype(str).tolist())
    chemistry[label_column] = dataset[label_column].to_numpy()
    valid = chemistry["L5_inchikey"].notna()

    rows = []
    for key, group in chemistry[valid].groupby("L5_inchikey", sort=False):
        first = group.index[0]
        labels = group[label_column]
        row = dataset.loc[first].to_dict()
        row.update({
            "smiles_original": dataset.loc[first, smiles_column],
            smiles_column: group.loc[first, "L4_tautomer"],
            label_column: int(labels.max()),
            "inchikey": key,
            "n_forms": len(group),
            "conflicting_label": bool(labels.nunique() > 1),
        })
        rows.append(row)
    standardized = pd.DataFrame(rows)

    summary = {
        "original_rows": len(dataset),
        "unparsable_by_rdkit": int((~valid).sum()),
        "distinct_substances": len(standardized),
        "rows_merged_as_duplicates": int(valid.sum() - len(standardized)),
        "groups_with_duplicates": int((standardized["n_forms"] > 1).sum()),
        "groups_with_conflicting_label": int(standardized["conflicting_label"].sum()),
        "positives_before": int(dataset[label_column].sum()),
        "positives_after": int(standardized[label_column].sum()),
    }
    return standardized, chemistry, summary


# --------------------------------------------------------------------------------------
# 2. Metrics at a fixed threshold
# --------------------------------------------------------------------------------------

def metrics(y, proba, threshold=DEFAULT_THRESHOLD):
    """The eight metrics of the study for one probability vector. The first six use the cut at
    ``threshold``; ROC-AUC and average precision do not depend on a threshold."""
    y = np.asarray(y)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    two_classes = len(np.unique(y)) > 1
    return {
        "accuracy": accuracy_score(y, pred),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "roc_auc": roc_auc_score(y, proba) if two_classes else np.nan,
        "average_precision": average_precision_score(y, proba) if two_classes else np.nan,
    }


def recall_at_k(y, scores, k):
    """Fraction of the positives that appear among the ``k`` highest-scoring compounds."""
    y = np.asarray(y)
    k = min(k, len(y))
    top = np.argsort(-np.asarray(scores), kind="mergesort")[:k]
    return float(y[top].sum() / y.sum()) if y.sum() else np.nan


def screening_metrics(y, scores, ks=SCREENING_K):
    """Screening metrics, all threshold-free: Precision@K, Recall@K and BEDROC (alpha = 20)."""
    output = {"compounds": len(y), "positives": int(np.sum(y)), "prevalence": float(np.mean(y))}
    for k in ks:
        output[f"Precision@{k}"] = precision_at_k(y, scores, k)
        output[f"Recall@{k}"] = recall_at_k(y, scores, k)
    output["BEDROC (α=20)"] = bedroc(y, scores, alpha=20.0)
    return output


def shared_scaffolds(folds, groups):
    """Per fold, how many scaffold groups appear in train and test at the same time."""
    groups = np.asarray(groups)
    return pd.DataFrame([{"seed": seed, "fold": k,
                          "scaffolds in test": len(set(groups[te])),
                          "scaffolds in train and test": len(set(groups[tr]) & set(groups[te]))}
                         for seed, k, tr, te in folds])


# --------------------------------------------------------------------------------------
# 3. Individual models: nested inner CV per fold
# --------------------------------------------------------------------------------------

def inner_folds(y_train, train_groups, protocol, seed, n_splits=5):
    """Inner folds of the training part, under the same criterion as the outer fold."""
    empty = np.zeros(len(y_train))
    if protocol == "scaffold":
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(empty, y_train, groups=train_groups))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(empty, y_train))


def fit_one_member(X, y, tr, te, inner_splits, C_grid, binary_, n_jobs=1):
    """One representation fitted in one outer fold.

    The inner CV, on the training part only, selects C by ROC-AUC and returns the inner
    out-of-fold probabilities that feed the ensembles. The fold's model is then fitted on the
    whole training part and predicts both train and test.
    """
    inner = inner_oof_and_C(X[tr], y[tr], inner_splits, C_grid, binary=binary_, n_jobs=n_jobs)
    model = build_pipeline(binary=binary_, C=inner["C"]).fit(X[tr], y[tr])
    return {"C": inner["C"], "inner_auc": inner["inner_auc"],
            "inner_oof": inner["inner_oof"],
            "train_proba": model.predict_proba(X[tr])[:, 1],
            "test_proba": model.predict_proba(X[te])[:, 1]}


def fit_members(X, y, folds, names, binary_set, C_grid, checkpoints, protocol,
                  groups=None, n_jobs=1, prefix=None):
    """Every individual model of one protocol, checkpointed per (fold, representation).

    ``prefix`` keeps the checkpoints of different datasets apart (default: the protocol name).
    """
    records = {}
    for seed, k, tr, te in folds:
        inner_splits = inner_folds(y[tr], None if groups is None else groups[tr], protocol, seed)
        for name in names:
            key = f"member_{prefix or protocol}_seed{seed}_fold{k}_{name}"
            records[(seed, k, name)] = checkpoints.run_or_load(
                key,
                lambda name=name: fit_one_member(X[name], y, tr, te, inner_splits, C_grid,
                                                 name in binary_set, n_jobs=n_jobs),
                verbose=False)
    return records


def metrics_table(y, folds, probas, protocol, dataset):
    """Train and test metrics per fold. ``probas[(seed, k, model)]`` holds the keys
    ``train_proba`` and ``test_proba``."""
    rows = []
    for (seed, k, model), record in probas.items():
        tr, te = folds[(seed, k)]
        for part, idx, key in (("train", tr, "train_proba"), ("test", te, "test_proba")):
            rows.append({"dataset": dataset, "protocol": protocol, "model": model, "seed": seed, "fold": k,
                           "part": part, **metrics(y[idx], record[key])})
    return pd.DataFrame(rows)


def mean_by_model(table):
    """Mean and standard deviation over the 15 folds, by dataset, protocol, model and part."""
    aggregated = table.groupby(["dataset", "protocol", "model", "part"])[METRICS].agg(["mean", "std"])
    aggregated.columns = [f"{m}_{statistic}" for m, statistic in aggregated.columns]
    return aggregated.reset_index()


def mean_sd(mean_, sd_, places=4):
    """The text "mean ± sd". Accepts scalars or Series sharing an index."""
    if isinstance(mean_, pd.Series):
        return pd.Series([mean_sd(m, d, places) for m, d in zip(mean_, sd_.loc[mean_.index])],
                         index=mean_.index)
    return f"{mean_:.{places}f} ± {sd_:.{places}f}"


def format_sd(table, places=4):
    """Joins each column ``X`` with its ``X sd`` into a single "mean ± sd" text."""
    output = table.copy()
    for column in [c for c in table.columns if f"{c} sd" in table.columns]:
        output[column] = mean_sd(table[column], table[f"{column} sd"], places)
    return output.drop(columns=[c for c in table.columns if str(c).endswith(" sd")])


def compare(mean_, column, a, b, metrics_=("f1", "roc_auc"), part="test"):
    """Mean and sd over the 15 folds of every model at two values of ``column`` (protocol or
    dataset), side by side, and the difference ``b - a``. ``mean_`` comes from ``mean_by_model``
    already filtered on the other axis. The folds of the two sides are not paired, so the sd of
    Δ is the combination of both, √(sd_a² + sd_b²)."""
    selected_rows = mean_[mean_["part"] == part]
    output = {}
    for m in metrics_:
        wide = selected_rows.pivot(index="model", columns=column, values=f"{m}_mean")
        sd_ = selected_rows.pivot(index="model", columns=column, values=f"{m}_std")
        name = METRIC_NAMES[m]
        output[f"{name} ({a})"] = wide[a]
        output[f"{name} ({a}) sd"] = sd_[a]
        output[f"{name} ({b})"] = wide[b]
        output[f"{name} ({b}) sd"] = sd_[b]
        output[f"Δ {name} ({b} − {a})"] = wide[b] - wide[a]
        output[f"Δ {name} ({b} − {a}) sd"] = np.sqrt(sd_[a] ** 2 + sd_[b] ** 2)
    return pd.DataFrame(output)


# --------------------------------------------------------------------------------------
# 4. Ensembles and greedy selection, decided from the training part of the fold alone
# --------------------------------------------------------------------------------------

def greedy_selection(inner_matrix, y_train, names, threshold=DEFAULT_THRESHOLD):
    """Greedy forward selection for soft voting, maximizing F1 at the fixed threshold.

    At each step it adds the representation that, together with those already chosen, gives the
    highest F1 of the averaged inner out-of-fold probabilities. The best step along the path is
    kept; ties go to the smaller subset. Uses the training part of the fold only.
    """
    columns = {name: inner_matrix[:, j] for j, name in enumerate(names)}
    remaining, chosen, path = list(names), [], []
    while remaining:
        scored = []
        for candidate in remaining:
            mean_ = np.mean([columns[m] for m in chosen + [candidate]], axis=0)
            scored.append((f1_score(y_train, (mean_ >= threshold).astype(int), zero_division=0),
                              candidate))
        f1, candidate = max(scored, key=lambda pair: pair[0])
        chosen.append(candidate)
        remaining.remove(candidate)
        path.append({"n_members": len(chosen), "added": candidate,
                        "members": tuple(chosen), "f1_inner": float(f1)})
    best = max(path, key=lambda step: (step["f1_inner"], -step["n_members"]))
    return list(best["members"]), pd.DataFrame(path)


def fit_meta_model(random_state=RANDOM_STATE):
    """The stacking meta-model: L2 logistic regression over the member probabilities."""
    return LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced",
                              random_state=random_state)


def fold_ensembles(y, tr, fold_records, names, seed):
    """The four ensembles of one outer fold.

    For each ensemble it returns the train and test probabilities plus an inner score, computed
    on the training part without looking at the test fold, later used to pick a threshold:

    * soft voting: the mean of the member probabilities;
    * weighted soft voting: that mean weighted by each member's inner ROC-AUC;
    * stacking: a meta-model fitted on the members' inner out-of-fold probabilities;
    * greedy selection: soft voting over the subset chosen on the inner probabilities.
    """
    inner_oof_matrix = np.column_stack([fold_records[n]["inner_oof"] for n in names])
    train_ = np.column_stack([fold_records[n]["train_proba"] for n in names])
    test_ = np.column_stack([fold_records[n]["test_proba"] for n in names])
    y_tr = y[tr]

    weights = np.array([fold_records[n]["inner_auc"] for n in names])
    weights = weights / weights.sum()

    meta = fit_meta_model().fit(inner_oof_matrix, y_tr)
    cv_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    meta_inner = cross_val_predict(clone(fit_meta_model()), inner_oof_matrix, y_tr, cv=cv_meta,
                                     method="predict_proba")[:, 1]

    subset, path = greedy_selection(inner_oof_matrix, y_tr, names)
    columns = [names.index(n) for n in subset]

    return {
        "Soft voting": {"train_proba": train_.mean(axis=1), "test_proba": test_.mean(axis=1),
                        "inner_score": inner_oof_matrix.mean(axis=1)},
        "Weighted soft voting (ROC-AUC)": {
            "train_proba": train_ @ weights, "test_proba": test_ @ weights,
            "inner_score": inner_oof_matrix @ weights, "weights": dict(zip(names, weights))},
        "Stacking": {"train_proba": meta.predict_proba(train_)[:, 1],
                     "test_proba": meta.predict_proba(test_)[:, 1],
                     "inner_score": meta_inner,
                     "coefficients": dict(zip(names, meta.coef_.ravel()))},
        "Greedy (F1)": {"train_proba": train_[:, columns].mean(axis=1),
                           "test_proba": test_[:, columns].mean(axis=1),
                           "inner_score": inner_oof_matrix[:, columns].mean(axis=1),
                           "members": subset, "path": path},
    }


def selection_frequency(ensembles, names):
    """In how many folds each representation entered the greedy ensemble."""
    counts = {n: 0 for n in names}
    total = 0
    for (seed, k, model), record in ensembles.items():
        if model == "Greedy (F1)":
            total += 1
            for n in record["members"]:
                counts[n] += 1
    return (pd.Series(counts, name="folds").sort_values(ascending=False).to_frame()
            .assign(frequency=lambda d: d["folds"] / total))


# --------------------------------------------------------------------------------------
# 5. Diversity
# --------------------------------------------------------------------------------------

def oof_hits(y, folds, records, names, seed, threshold=DEFAULT_THRESHOLD):
    """Hit/miss vector (at the fixed threshold) of each member, assembled from the test
    predictions of one repetition: every compound is predicted once, by a model that has not
    seen it."""
    y = np.asarray(y)
    proba = {n: np.full(len(y), np.nan) for n in names}
    for (s, k), (tr, te) in folds.items():
        if s != seed:
            continue
        for n in names:
            proba[n][te] = records[(s, k, n)]["test_proba"]
    return {n: (proba[n] >= threshold).astype(int) == y for n in names}


DIVERSITY_MEASURES = {
    "Double-fault": double_fault_measure,
    "Q-statistic": q_statistic,
    "Disagreement": disagreement_measure,
    "Correlation coefficient": correlation_measure,
}


# --------------------------------------------------------------------------------------
# 6. Threshold sweep
# --------------------------------------------------------------------------------------

def threshold_sweep(y, folds, probas, grid=THRESHOLD_GRID):
    """Mean-F1 curves per threshold, on the inner out-of-fold scores and on test, plus the
    selected threshold.

    ``probas[(seed, k)]`` holds ``test_proba`` and ``inner_score``. There is one threshold per
    model: the peak of the mean out-of-fold curve over the 15 folds. The test fold never enters
    the choice, but the out-of-fold scores of one fold come from a training set containing test
    compounds of other folds, so there is mild leakage (a single scalar). The test curve is for
    display only. Returns the F1 per fold and threshold, the curves (mean and sd over the 15
    folds) and the selected threshold.
    """
    rows = []
    for (seed, k), record in probas.items():
        tr, te = folds[(seed, k)]
        for threshold in grid:
            rows.append({
                "seed": seed, "fold": k, "threshold": threshold,
                "f1_oof": f1_score(y[tr], (record["inner_score"] >= threshold).astype(int),
                                   zero_division=0),
                "f1_test": f1_score(y[te], (record["test_proba"] >= threshold).astype(int),
                                     zero_division=0)})
    per_fold = pd.DataFrame(rows)
    curves = per_fold.groupby("threshold")[["f1_oof", "f1_test"]].agg(["mean", "std"])
    curves.columns = [f"{c}_{statistic}" for c, statistic in curves.columns]
    return per_fold, curves, float(curves["f1_oof_mean"].idxmax())


# --------------------------------------------------------------------------------------
# 7. Final models on the whole dataset
# --------------------------------------------------------------------------------------

def fit_final_models(X, y, names, binary_set, C_grid, groups, seed=RANDOM_STATE, n_jobs=1):
    """Individual models and ensembles fitted on all the data.

    The C of each representation and the out-of-fold probabilities that feed the weights, the
    meta-model and the greedy subset come from a 5-fold scaffold CV over the whole dataset.
    """
    y = np.asarray(y)
    inner_splits = inner_folds(y, groups, "scaffold", seed)
    fitted_members, oof, auc = {}, {}, {}
    for name in names:
        inner = inner_oof_and_C(X[name], y, inner_splits, C_grid, binary=name in binary_set,
                                  n_jobs=n_jobs)
        fitted_members[name] = build_pipeline(binary=name in binary_set,
                                           C=inner["C"]).fit(X[name], y)
        oof[name], auc[name] = inner["inner_oof"], inner["inner_auc"]
    matrix = np.column_stack([oof[n] for n in names])
    weights = np.array([auc[n] for n in names])
    subset, path = greedy_selection(matrix, y, names)
    return {
        "members": fitted_members,
        "C": {n: fitted_members[n].named_steps["clf"].C for n in names},
        "roc_auc_weights": dict(zip(names, weights / weights.sum())),
        "meta_model": fit_meta_model().fit(matrix, y),
        "greedy_members": subset,
        "greedy_path": path,
        "names": list(names),
    }


def encoders_of_model(final, model):
    """The representations a final model needs in order to predict."""
    if model == "Greedy (F1)":
        return list(final["greedy_members"])
    if model in ENSEMBLES:
        return list(final["names"])
    return [model]


def predict_with_model(final, X, model):
    """Probabilities of one final model (individual or ensemble) for new compounds.

    ``X`` only needs the representations listed by ``encoders_of_model``."""
    names = encoders_of_model(final, model)
    matrix = np.column_stack([final["members"][n].predict_proba(X[n])[:, 1] for n in names])
    if model == "Weighted soft voting (ROC-AUC)":
        return matrix @ np.array([final["roc_auc_weights"][n] for n in names])
    if model == "Stacking":
        return final["meta_model"].predict_proba(matrix)[:, 1]
    return matrix.mean(axis=1)


def predict_with_final_models(final, X):
    """Probabilities of every final model for new compounds (a dict of embeddings)."""
    names = final["names"]
    matrix = np.column_stack([final["members"][n].predict_proba(X[n])[:, 1] for n in names])
    weights = np.array([final["roc_auc_weights"][n] for n in names])
    columns = [names.index(n) for n in final["greedy_members"]]
    output = pd.DataFrame(matrix, columns=names)
    output["Soft voting"] = matrix.mean(axis=1)
    output["Weighted soft voting (ROC-AUC)"] = matrix @ weights
    output["Stacking"] = final["meta_model"].predict_proba(matrix)[:, 1]
    output["Greedy (F1)"] = matrix[:, columns].mean(axis=1)
    return output
