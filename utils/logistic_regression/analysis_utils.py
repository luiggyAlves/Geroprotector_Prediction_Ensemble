"""The analyses that address five of the limitations discussed in the write-up.

Nothing in the protocol changes: the same 18 encoders, the same 15 folds and the same 1,080
member checkpoints are reused. What this module adds:

* **Calibrated stacking** (limitation vi): isotonic and Platt calibration fitted inside the
  fold, on the meta-model's own inner out-of-fold probabilities.
* **Greedy selection by average precision** (limitation v): a threshold-free criterion, which
  removes the circularity between member selection and the threshold study.
* **Scaffold bootstrap and paired tests** (limitation i): confidence intervals and comparisons,
  built on the bootstrap helpers in `core_utils`.
* **InChIKey collapse** (limitation iii): rescores the unstandardized predictions with one
  prediction per substance, to test whether the standardization drop is a counting effect.
* **Label-conflict ablation** (limitation iv): the three policies (positive, removed, negative)
  for the 82 substances whose forms disagreed on the label after the InChIKey merge.

The whole pipeline comes from protocol_utils and the bootstrap functions from core_utils.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from core_utils import (all_metrics, corrected_resampled_ttest,  # noqa: F401
                         group_bootstrap_ci, group_bootstrap_indices,
                         paired_group_bootstrap)
from protocol_utils import (ENSEMBLES, THRESHOLD_GRID, SCREENING_K, DEFAULT_THRESHOLD,  # noqa: F401
                         DIVERSITY_MEASURES, METRICS, METRIC_NAMES, RANDOM_STATE,
                         Checkpoints, oof_hits, bedroc, build_pipeline, chemistry_table,
                         compare, correlation_measure, disagreement_measure,
                         double_fault_measure, duplicate_audit, encoders_of_model,
                         threshold_sweep, external_cuts, inner_folds, format_sd,
                         inner_oof_and_C, log_external_read, mean_sd, mean_by_model,
                         metrics, screening_metrics, standardize_dataset, pairwise_matrix,
                         precision_at_k, predict_with_final_models, predict_with_model, q_statistic,
                         recall_at_k, repeated_random_folds, repeated_scaffold_folds,
                         fit_members, shared_scaffolds, greedy_selection,
                         metrics_table, fit_final_models)

# The full ensemble set: the four original ones plus the two calibrated stackings and the
# average-precision greedy selection. "Greedy (F1)" remains the F1-at-0.5 variant.
LATER_ENSEMBLES = ["Stacking (isotonic)", "Stacking (Platt)", "Greedy (AP)"]
ALL_ENSEMBLES = list(ENSEMBLES) + LATER_ENSEMBLES

CONFLICT_POLICIES = ["positive", "removed", "negative"]

# Wide grid for the threshold chosen inside the fold. The 0.40-0.60 grid only suits models
# whose probabilities already sit near 0.5; a calibrated model at 17% prevalence has its
# optimum far below that.
WIDE_THRESHOLD_GRID = np.round(np.arange(0.05, 0.9501, 0.01), 2)


# --------------------------------------------------------------------------------------
# 1. Calibrated stacking (limitation vi)
# --------------------------------------------------------------------------------------

def fit_meta_model(random_state=RANDOM_STATE):
    """The stacking meta-model: L2 logistic regression over the member probabilities."""
    return LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced",
                              random_state=random_state)


def calibrate(inner_score_, y_train, method):
    """A calibrator fitted on the meta-model's inner out-of-fold probabilities.

    The meta-model uses ``class_weight="balanced"``, which shifts its probabilities upward: at
    threshold 0.5 it calls positive too often, with high recall and low precision. The
    calibrator learns the monotone map that turns the score back into a probability, and is
    fitted on the training part alone (on the inner predictions, which the final meta-model did
    not see while each subfold was fitted).

    * ``"isotonic"``: isotonic regression, flexible and with no functional form;
    * ``"platt"``: one-variable logistic regression (a sigmoid scale).

    Returns a function that applies the calibration to a vector of scores.
    """
    inner_score_ = np.asarray(inner_score_, dtype=float)
    y_train = np.asarray(y_train)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(inner_score_, y_train)
        return lambda scores: np.clip(model.predict(np.asarray(scores, dtype=float)), 0, 1)
    if method == "platt":
        model = LogisticRegression(C=1e10, max_iter=5000, solver="lbfgs")
        model.fit(inner_score_.reshape(-1, 1), y_train)
        return lambda scores: model.predict_proba(
            np.asarray(scores, dtype=float).reshape(-1, 1))[:, 1]
    raise ValueError(f"unknown calibration method: {method}")


# --------------------------------------------------------------------------------------
# 2. Threshold-free greedy selection (limitation v)
# --------------------------------------------------------------------------------------

def greedy_selection_ap(inner_matrix, y_train, names):
    """Greedy forward selection maximizing the average precision of the averaged probabilities.

    The F1 variant maximizes F1 at threshold 0.5, which leaves the inner score optimized at
    that point and makes the threshold study circular for that ensemble. Average precision
    needs no threshold and is the ranking metric closest to screening use at 17% prevalence.
    The rest of the procedure is identical: only the training part of the fold is used and, on
    a tie, the smaller subset wins.
    """
    columns = {name: inner_matrix[:, j] for j, name in enumerate(names)}
    remaining, chosen, path = list(names), [], []
    while remaining:
        scored = []
        for candidate in remaining:
            mean_ = np.mean([columns[m] for m in chosen + [candidate]], axis=0)
            scored.append((average_precision_score(y_train, mean_), candidate))
        pr_auc, candidate = max(scored, key=lambda pair: pair[0])
        chosen.append(candidate)
        remaining.remove(candidate)
        path.append({"n_members": len(chosen), "added": candidate,
                        "members": tuple(chosen), "inner_ap": float(pr_auc)})
    best = max(path, key=lambda step: (step["inner_ap"], -step["n_members"]))
    return list(best["members"]), pd.DataFrame(path)


# --------------------------------------------------------------------------------------
# 3. The fold ensembles, including the three later ones
# --------------------------------------------------------------------------------------

def fold_ensembles(y, tr, fold_records, names, seed):
    """The seven ensembles of one outer fold.

    The four basic ones (soft voting, weighted soft voting, stacking and F1 greedy selection)
    plus the two calibrated stackings and the average-precision greedy selection. Every
    decision comes from the inner out-of-fold probabilities of the training part; the test fold
    enters no choice.
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
    meta_train = meta.predict_proba(train_)[:, 1]
    meta_test = meta.predict_proba(test_)[:, 1]

    subset, path = greedy_selection(inner_oof_matrix, y_tr, names)
    columns = [names.index(n) for n in subset]
    subset_ap, path_ap = greedy_selection_ap(inner_oof_matrix, y_tr, names)
    columns_ap = [names.index(n) for n in subset_ap]

    output = {
        "Soft voting": {"train_proba": train_.mean(axis=1), "test_proba": test_.mean(axis=1),
                        "inner_score": inner_oof_matrix.mean(axis=1)},
        "Weighted soft voting (ROC-AUC)": {
            "train_proba": train_ @ weights, "test_proba": test_ @ weights,
            "inner_score": inner_oof_matrix @ weights, "weights": dict(zip(names, weights))},
        "Stacking": {"train_proba": meta_train, "test_proba": meta_test,
                     "inner_score": meta_inner,
                     "coefficients": dict(zip(names, meta.coef_.ravel()))},
        "Greedy (F1)": {"train_proba": train_[:, columns].mean(axis=1),
                           "test_proba": test_[:, columns].mean(axis=1),
                           "inner_score": inner_oof_matrix[:, columns].mean(axis=1),
                           "members": subset, "path": path},
        "Greedy (AP)": {"train_proba": train_[:, columns_ap].mean(axis=1),
                                    "test_proba": test_[:, columns_ap].mean(axis=1),
                                    "inner_score": inner_oof_matrix[:, columns_ap].mean(axis=1),
                                    "members": subset_ap, "path": path_ap},
    }
    # Both calibrated stackings reuse the same meta-model: only the final map changes, so the
    # ordering — and therefore ROC-AUC and average precision — is identical to plain stacking.
    for method, name in (("isotonic", "Stacking (isotonic)"), ("platt", "Stacking (Platt)")):
        calibrator = calibrate(meta_inner, y_tr, method)
        output[name] = {"train_proba": calibrator(meta_train),
                       "test_proba": calibrator(meta_test),
                       "inner_score": calibrator(meta_inner),
                       "calibration_method": method}
    return output


def in_fold_threshold(y, folds, records, models, grid=WIDE_THRESHOLD_GRID,
                   default_threshold=DEFAULT_THRESHOLD):
    """Test metrics at threshold 0.5 and at the threshold chosen inside each fold.

    In every fold the threshold is the one that maximizes F1 on the inner out-of-fold
    probabilities of the training part, over a wide grid; the test fold does not enter the
    choice. Calibrating a model and still reading it at 0.5 settles nothing: calibration
    shrinks the probabilities toward the prevalence (17%), so at 0.5 almost nothing is called
    positive and recall collapses. It is calibration together with an in-fold threshold that
    answers the question.
    """
    y = np.asarray(y)
    rows = []
    for (seed, k), (tr, te) in folds.items():
        for model in models:
            record = records[(seed, k, model)]
            inner, test_ = record["inner_score"], record["test_proba"]
            inner_f1 = [f1_score(y[tr], (inner >= threshold).astype(int), zero_division=0)
                          for threshold in grid]
            selected = float(grid[int(np.argmax(inner_f1))])
            rows.append({
                "model": model, "seed": seed, "fold": k, "selected threshold": selected,
                **{f"{k_}: 0.5": v for k_, v in metrics(y[te], test_, default_threshold).items()},
                **{f"{k_}: in-fold threshold": v
                   for k_, v in metrics(y[te], test_, selected).items()},
            })
    return pd.DataFrame(rows)


def selection_frequency(ensembles, names, model="Greedy (F1)"):
    """In how many folds each representation entered one of the greedy ensembles."""
    counts = {n: 0 for n in names}
    total = 0
    for (seed, k, name), record in ensembles.items():
        if name == model:
            total += 1
            for n in record["members"]:
                counts[n] += 1
    return (pd.Series(counts, name="folds").sort_values(ascending=False).to_frame()
            .assign(frequency=lambda d: d["folds"] / total))


# --------------------------------------------------------------------------------------
# 4. Scaffold bootstrap and paired tests (limitation i)
# --------------------------------------------------------------------------------------

def oof_of_one_repetition(y, folds, records, models, seed=RANDOM_STATE):
    """Assembles the out-of-fold prediction vector of one repetition, per model.

    Within a repetition the 5 folds cover the whole dataset, so every compound is predicted
    exactly once, by a model that has not seen it. That vector is what the bootstrap resamples,
    whole scaffolds at a time.
    """
    y = np.asarray(y)
    proba = {m: np.full(len(y), np.nan) for m in models}
    for (s, k), (tr, te) in folds.items():
        if s != seed:
            continue
        for m in models:
            proba[m][te] = records[(s, k, m)]["test_proba"]
    missing = {m: int(np.isnan(v).sum()) for m, v in proba.items() if np.isnan(v).any()}
    assert not missing, f"incomplete out-of-fold predictions: {missing}"
    return proba


def intervals_table(y, proba, groups, models, metrics_=None, n_boot=2000,
                      threshold=DEFAULT_THRESHOLD, seed=RANDOM_STATE):
    """95% CI of each model, resampling whole scaffolds."""
    metrics_ = metrics_ or ["roc_auc", "average_precision", "f1", "matthews_corrcoef",
                              "bedroc_20", "precision_at_100"]
    rows = []
    for m in models:
        table = group_bootstrap_ci(y, proba[m], groups, threshold, metrics=metrics_,
                                    n_boot=n_boot, seed=seed)
        rows.append(table.assign(model=m))
    return pd.concat(rows, ignore_index=True)[
        ["model", "metric", "point", "ci_low", "ci_high", "n_replicates"]]


def paired_table(y, proba, groups, challengers, reference, metrics_=None, n_boot=2000,
                   threshold=DEFAULT_THRESHOLD, seed=RANDOM_STATE):
    """Difference challenger − reference, with the same resample applied to both models."""
    metrics_ = metrics_ or ["roc_auc", "average_precision", "f1", "matthews_corrcoef",
                              "bedroc_20"]
    rows = []
    for m in challengers:
        table = paired_group_bootstrap(y, proba[m], proba[reference], groups, threshold, threshold,
                                        metrics=metrics_, n_boot=n_boot, seed=seed)
        rows.append(table.assign(model=m, reference=reference))
    return pd.concat(rows, ignore_index=True)[
        ["model", "reference", "metric", "A", "B", "difference", "ci_low", "ci_high",
         "p_two_sided", "excludes_zero"]]


def nadeau_bengio_table(per_fold, challengers, reference, metrics_=None, part="test"):
    """Corrected resampled t-test (Nadeau-Bengio) over the 15 folds.

    The folds of a repeated CV share training data, so the ordinary paired t-test is
    anticonservative. The correction replaces the 1/n variance factor by 1/n + n_test/n_train.
    With 5-fold splits, n_test/n_train = 1/4, so the factor goes from 1/15 to 0.3167.
    """
    metrics_ = metrics_ or ["f1", "roc_auc", "average_precision", "mcc"]
    table = per_fold.query("part == @part").set_index(["model", "seed", "fold"])
    rows = []
    for m in challengers:
        for metric_ in metrics_:
            a = table.loc[m, metric_].sort_index()
            b = table.loc[reference, metric_].sort_index()
            result = corrected_resampled_ttest(a.to_numpy(), b.to_numpy(),
                                                  n_train=4, n_test=1)
            rows.append({"model": m, "reference": reference, "metric": metric_,
                           "mean Δ": result["mean_difference"],
                           "corrected p": result["p_value"],
                           "uncorrected p": result["naive_p_value"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 5. InChIKey collapse (limitation iii)
# --------------------------------------------------------------------------------------

def collapse_by_inchikey(y, proba, keys_):
    """Aggregates the unstandardized predictions into one prediction per substance.

    In the unstandardized dataset a substance with several forms is counted several times in
    the test fold. If the substances with more forms are the easier ones, F1 is inflated
    without the model being any better. Collapsing by InChIKey (the mean of the probabilities,
    the label by the same "positive if any form is positive" rule used by standardization)
    removes that counting effect without touching the model. If the Δ against the standardized
    dataset disappears, the drop was counting.

    Rows without an InChIKey are left out. Returns the label and the probability per substance.
    """
    table = pd.DataFrame({"key": np.asarray(keys_), "y": np.asarray(y)})
    for name, vector in proba.items():
        table[name] = np.asarray(vector)
    table = table[table["key"].notna()]
    aggregated = table.groupby("key", sort=False).agg(
        {"y": "max", **{name: "mean" for name in proba}})
    return aggregated["y"].to_numpy(), {n: aggregated[n].to_numpy() for n in proba}


# --------------------------------------------------------------------------------------
# 6. Label-conflict ablation (limitation iv)
# --------------------------------------------------------------------------------------

def apply_conflict_policy(dataset, policy):
    """Applies one of the three policies to the substances with conflicting labels.

    The InChIKey merge joins forms of the same substance that arrived with different labels;
    the default resolves them all as positive. The three policies measure how much of that
    choice shows up in the results:

    * ``"positive"``: the default rule (nothing changes);
    * ``"removed"``: the conflicting substances leave the dataset;
    * ``"negative"``: they become negatives.

    Returns the dataset with the policy applied and the kept positions, so that embeddings and
    scaffold groups can be sliced in the same order.
    """
    conflicting = dataset["conflicting_label"].to_numpy().astype(bool)
    if policy == "positive":
        return dataset.copy(), np.arange(len(dataset))
    if policy == "negative":
        output = dataset.copy()
        output.loc[conflicting, "label"] = 0
        return output, np.arange(len(dataset))
    if policy == "removed":
        kept = np.flatnonzero(~conflicting)
        return dataset.iloc[kept].reset_index(drop=True), kept
    raise ValueError(f"unknown policy: {policy}")
