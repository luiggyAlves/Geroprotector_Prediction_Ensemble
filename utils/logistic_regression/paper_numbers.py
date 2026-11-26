"""Checks every number in the paper against the stored results in results/tables/.

This is the short path to reproduction: it runs in seconds, with no embeddings and no model
fitting, because it reads the tables the benchmark notebook wrote. To rebuild those tables from
scratch, run the notebook (see the README).

Each line prints the value in the paper, the value computed from the files, and a verdict.
Rounding is half-up, as in the text: Python would round 0.0215 down to 0.021, while the paper
writes 0.022.

Usage:
    python utils/logistic_regression/paper_numbers.py
"""

from __future__ import annotations

import os
import sys
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TABLES = os.path.join(ROOT, "results", "tables")
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, HERE)

import analysis_utils as U  # noqa: E402

PAPER_ENSEMBLE = "Greedy (AP)"
BEST_INDIVIDUAL = "MolFormer-XL"
checks: list[dict] = []


def rounded(value, places):
    """Half-up rounding, as the text of the paper uses."""
    return float(Decimal(str(float(value))).quantize(Decimal(f"1e-{places}"),
                                                     rounding=ROUND_HALF_UP))


def check(section, claim, expected, computed, places=3):
    """Records one check. `expected` None only prints the computed value."""
    ok = None if expected is None else rounded(computed, places) == expected
    checks.append({"section": section, "claim in the paper": claim, "paper": expected,
                   "computed": rounded(computed, places),
                   "verdict": {True: "ok", False: "MISMATCH", None: "—"}[ok]})


def table(name):
    return pd.read_csv(os.path.join(TABLES, name))


# ----------------------------------------------------------------------------- the datasets
def section_datasets():
    rows = [("§4.1 unstandardized dataset", "gerollm_reconstructed/primary_dataset_dedup.csv",
             4233, 727, 17.17),
            ("§4.1 standardized dataset", "gerollm_standardized/primary_standardized.csv",
             3752, 637, 16.98),
            ("§4.1 external set (raw)", "gerollm_reconstructed/external_test_set.csv",
             1557, 35, 2.25)]
    for label, path, n, positives, prevalence in rows:
        d = pd.read_csv(os.path.join(DATA, path))
        check(label, "compounds", n, len(d), 0)
        check(label, "positives", positives, int(d["label"].sum()), 0)
        check(label, "prevalence (%)", prevalence, 100 * d["label"].mean(), 2)
        if "n_forms" in d:
            check(label, "rows merged as duplicates", 481, int((d["n_forms"] - 1).sum()), 0)
            check(label, "groups of redundant forms", 411, int((d["n_forms"] > 1).sum()), 0)
            check(label, "groups with conflicting labels", 82,
                  int(d["conflicting_label"].sum()), 0)


# ------------------------------------------------------------------------ splits and leakage
def section_splits():
    v = table("scaffold_leakage_check.csv")
    scaffold = v[v["protocol"] == "scaffold split"]
    check("§3.3 scaffold split", "scaffolds in train and test at once", 0,
          int(scaffold["scaffolds in train and test"].sum()), 0)
    for dataset, expected in (("standardized dataset", 227), ("unstandardized dataset", 301)):
        sub = v[(v["protocol"] == "random stratified split") & (v["dataset"] == dataset)]
        check("§3.3 random split", f"scaffolds shared per fold ({dataset})", expected,
              sub["scaffolds in train and test"].mean(), 0)


# ------------------------------------------------------------- tables 1 and 2, and the 80
def means():
    individual = table("individual_metrics_mean.csv").query("part == 'test'")
    ensemble = table("ensemble_metrics_mean.csv").query("part == 'test'")
    return individual, ensemble


def section_tables():
    individual, ensemble = means()
    TABLE1 = {("unstandardized", "random"): (0.062, 0.040, 0.088, 0.079),
              ("standardized", "random"): (0.043, 0.036, 0.059, 0.058),
              ("unstandardized", "scaffold"): (0.026, 0.024, 0.056, 0.035),
              ("standardized", "scaffold"): (0.012, 0.017, 0.033, 0.017)}
    METRICS = ("f1", "roc_auc", "average_precision", "mcc")
    for (dataset, protocol), expected_row in TABLE1.items():
        ind = individual.query("dataset == @dataset and protocol == @protocol")
        best = ind.loc[ind["f1_mean"].idxmax()]
        ens = ensemble.query("dataset == @dataset and protocol == @protocol "
                             "and model == @PAPER_ENSEMBLE").iloc[0]
        for metric, expected in zip(METRICS, expected_row):
            check("Table 1", f"Δ{metric} ({dataset}, {protocol})", expected,
                  ens[f"{metric}_mean"] - best[f"{metric}_mean"])

    setting = "dataset == 'standardized' and protocol == 'scaffold'"
    TABLE2 = {BEST_INDIVIDUAL: ((0.423, 0.036), (0.735, 0.040), (0.428, 0.065)),
              PAPER_ENSEMBLE: ((0.435, 0.045), (0.752, 0.044), (0.461, 0.059))}
    for model, expected_row in TABLE2.items():
        source = individual if model == BEST_INDIVIDUAL else ensemble
        row = source.query(setting).query("model == @model").iloc[0]
        for metric, (mean, sd) in zip(("f1", "roc_auc", "average_precision"), expected_row):
            check("Table 2", f"{model} {metric} (mean)", mean, row[f"{metric}_mean"])
            check("Table 2", f"{model} {metric} (sd)", sd, row[f"{metric}_std"])

    # "ensembles won all 80 comparisons" (§4.2 and the abstract)
    negative = 0
    for dataset in ("unstandardized", "standardized"):
        for protocol in ("random", "scaffold"):
            ind = individual.query("dataset == @dataset and protocol == @protocol")
            best = ind.loc[ind["f1_mean"].idxmax()]
            for model in U.ENSEMBLES + [PAPER_ENSEMBLE]:
                row = ensemble.query("dataset == @dataset and protocol == @protocol "
                                     "and model == @model").iloc[0]
                negative += sum(row[f"{m}_mean"] - best[f"{m}_mean"] <= 0 for m in METRICS)
    check("§4.2", "ensemble-vs-best comparisons with Δ <= 0 (out of 80)", 0, negative, 0)

    # precision traded for recall (§4.2)
    greedy = ensemble.query(setting).query("model == 'Greedy (F1)'").iloc[0]
    best = individual.query(setting).query("model == @BEST_INDIVIDUAL").iloc[0]
    check("§4.2", "Greedy (F1) Δprecision", 0.035,
          greedy["precision_mean"] - best["precision_mean"])
    check("§4.2", "Greedy (F1) Δrecall", -0.047, greedy["recall_mean"] - best["recall_mean"])


# ------------------------------------------------------------------------------- bootstrap
def section_bootstrap():
    paired = table("paired_bootstrap_scaffold_standardized.csv").query("model == @PAPER_ENSEMBLE")
    EXPECTED = {"roc_auc": (0.016, 0.003, 0.027), "average_precision": (0.030, 0.004, 0.052),
                "f1": (0.017, -0.002, 0.036)}
    for metric, (difference, low, high) in EXPECTED.items():
        row = paired.query("metric == @metric").iloc[0]
        check("§4.3", f"paired Δ{metric}", difference, row["difference"])
        check("§4.3", f"CI of Δ{metric} (lower)", low, row["ci_low"])
        check("§4.3", f"CI of Δ{metric} (upper)", high, row["ci_high"])

    everything = table("paired_bootstrap_scaffold_standardized.csv")
    ranking = everything[everything["metric"].isin(["roc_auc", "average_precision"])]
    exclude_zero = sorted(ranking[ranking["excludes_zero"]]["model"].unique())
    check("§4.3", "models excluding zero on ROC-AUC/AP (only Greedy (AP))", 1,
          len(exclude_zero), 0)
    assert exclude_zero == [PAPER_ENSEMBLE], exclude_zero

    intervals = table("bootstrap_intervals_scaffold_standardized.csv")
    CONCLUSION = {(PAPER_ENSEMBLE, "roc_auc"): (0.754, 0.725, 0.784),
                  (PAPER_ENSEMBLE, "average_precision"): (0.461, 0.412, 0.513),
                  (BEST_INDIVIDUAL, "roc_auc"): (0.738, 0.707, 0.773),
                  (BEST_INDIVIDUAL, "average_precision"): (0.430, 0.376, 0.493)}
    for (model, metric), (point, low, high) in CONCLUSION.items():
        row = intervals.query("model == @model and metric == @metric").iloc[0]
        check("Conclusion", f"{model} {metric}", point, row["point"])
        check("Conclusion", f"{model} {metric} CI lower", low, row["ci_low"])
        check("Conclusion", f"{model} {metric} CI upper", high, row["ci_high"])


# -------------------------------------------------------------- threshold and calibration
def section_threshold():
    d = table("threshold_by_selection_criterion.csv")
    row = d.query("dataset == 'standardized dataset' and criterion == @PAPER_ENSEMBLE").iloc[0]
    check("§4.4", "out-of-fold peak threshold, Greedy (AP)", 0.54,
          row["selected threshold (out-of-fold peak)"], 2)
    check("§4.4", "test F1 at 0.5", 0.435, row["test F1 at 0.5"])
    check("§4.4", "test F1 at the selected threshold", 0.443,
          row["test F1 at the selected threshold"])
    gain = (d["test F1 at the selected threshold"] - d["test F1 at 0.5"]).max()
    check("§4.4", "largest F1 gain from moving the threshold", 0.008, gain)
    greedy_f1 = d.query("dataset == 'standardized dataset' and criterion == 'Greedy (F1)'").iloc[0]
    check("§4.4", "out-of-fold peak threshold, Greedy (F1)", 0.50,
          greedy_f1["selected threshold (out-of-fold peak)"], 2)

    in_fold = table("in_fold_threshold_scaffold_standardized.csv")
    f1 = in_fold.groupby("model")["f1: in-fold threshold"].mean()
    spread = f1[["Stacking", "Stacking (Platt)", "Stacking (isotonic)"]]
    # The paper gives no number here, only that the stackings "converge to the same F1".
    check("§4.4", "spread of the stacking F1 with the in-fold threshold (paper: ~0)", None,
          spread.max() - spread.min())


# ------------------------------------------------------------- across-model aggregations
def section_optimism():
    """The section that needs the 22-model set; see the README."""
    all_means = pd.concat([table("individual_metrics_mean.csv"),
                           table("ensemble_metrics_mean.csv")], ignore_index=True)
    every_model = sorted(all_means["model"].unique())
    sets = {22: [m for m in every_model if m not in U.LATER_ENSEMBLES], 25: every_model}
    rhos = {22: [], 25: []}

    # On the standardized dataset the paper writes +0.022, which is the 0.0215 the notebook
    # prints, rounded once more; the exact value is 0.021500, which is 0.021 at three places.
    # The check uses four places there, so that a rounding artifact is not read as a mismatch.
    EXPECTED_RANDOM = {"unstandardized": (0.042, 0.041, 3), "standardized": (0.0215, 0.024, 4)}
    for dataset, (d_f1, d_auc, places_f1) in EXPECTED_RANDOM.items():
        comparison = U.compare(all_means.query("dataset == @dataset"), "protocol",
                               "scaffold", "random")
        for n, models in sets.items():
            sub = comparison.loc[[m for m in comparison.index if m in models]]
            rhos[n].append(spearmanr(sub["F1 (scaffold)"], sub["F1 (random)"]).statistic)
            check("§4.5", f"mean Δ F1 random − scaffold ({dataset}, {n} models)",
                  d_f1 if n == 22 else None, sub["Δ F1 (random − scaffold)"].mean(),
                  places_f1 if n == 22 else 3)
            check("§4.5", f"mean Δ ROC-AUC random − scaffold ({dataset}, {n} models)",
                  d_auc if n == 22 else None, sub["Δ ROC-AUC (random − scaffold)"].mean())

    for protocol in ("scaffold", "random"):
        comparison = U.compare(all_means.query("protocol == @protocol"), "dataset",
                               "unstandardized", "standardized")
        for n, models in sets.items():
            sub = comparison.loc[[m for m in comparison.index if m in models]]
            rhos[n].append(spearmanr(sub["F1 (unstandardized)"],
                                     sub["F1 (standardized)"]).statistic)
            if protocol == "scaffold":
                delta = sub["Δ F1 (standardized − unstandardized)"]
                check("§4.5", f"mean Δ F1 standardized − unstandardized (scaffold, {n} models)",
                      -0.016 if n == 22 else None, delta.mean())
                check("§4.5", f"models declining under standardization ({n} models)",
                      19 if n == 22 else None, int((delta < 0).sum()), 0)
    for n in (22, 25):
        check("§4.2", f"minimum Spearman across settings ({n} models)",
              0.948 if n == 22 else None, min(rhos[n]))

    # Graph-FP: its rank and the largest train-test gap
    ranking = table("ranking_standardized_scaffold.csv")
    if "rank (paper's 22)" in ranking:
        check("§4.2", "rank of Graph-FP among the 22", 15,
              float(ranking.loc[ranking["model"] == "Graph-FP", "rank (paper's 22)"].iloc[0]), 0)
    individual = table("individual_metrics_mean.csv").query(
        "dataset == 'standardized' and protocol == 'scaffold'")
    gap = individual.pivot_table(index="model", columns="part", values="f1_mean")
    gap["gap"] = gap["train"] - gap["test"]
    check("§4.2", "train-test gap of Graph-FP", 0.211, gap.loc["Graph-FP", "gap"])
    check("§4.2", "Graph-FP has the largest gap (1 = yes)", 1,
          int(gap["gap"].idxmax() == "Graph-FP"), 0)


# ------------------------------------------------- collapse, ablation and the external set
def section_collapse_ablation_external():
    collapse = table("inchikey_collapse_delta.csv").set_index("model")
    row = collapse.loc[BEST_INDIVIDUAL]
    against_rows = float(str(row["Δ F1 (standardized − rows)"]).split("±")[0])
    against_collapsed = float(str(row["Δ F1 (standardized − collapsed)"]).split("±")[0])
    check("§4.5", "Δ F1 of MolFormer-XL against the rows", -0.027, against_rows)
    check("§4.5", "Δ F1 of MolFormer-XL against the collapsed dataset", -0.019, against_collapsed)
    check("§4.5", "share of the drop explained by counting", 0.30,
          (against_rows - against_collapsed) / against_rows, 2)

    spearman = table("label_ablation_spearman.csv").select_dtypes("number")
    check("§4.6", "minimum Spearman between label policies", 0.977, spearman.min().min())
    gain = table("label_ablation_ensemble_gain.csv")
    check("§4.6", "policies where the ensemble loses (out of 12)", 0,
          int((gain["Δ (ensemble − individual)"] <= 0).sum()), 0)

    external = table("external_screening.csv").query(
        "`external set` == 'standardized external set' and cut == 'full'").set_index("model")
    check("§4.7", "compounds in the standardized external set", 1057,
          int(external.loc[BEST_INDIVIDUAL, "compounds"]), 0)
    check("§4.7", "positives in the standardized external set", 32,
          int(external.loc[BEST_INDIVIDUAL, "positives"]), 0)
    check("§4.7", "prevalence of the standardized external set (%)", 3.03,
          100 * external.loc[BEST_INDIVIDUAL, "prevalence"], 2)
    for column, (ens, ind), places in (("Precision@100", (0.15, 0.13), 2),
                                       ("Recall@100", (0.469, 0.406), 3),
                                       ("BEDROC (α=20)", (0.346, 0.306), 3)):
        check("§4.7", f"{column} of the ensemble", ens,
              external.loc["Greedy (F1)", column], places)
        check("§4.7", f"{column} of MolFormer-XL", ind,
              external.loc[BEST_INDIVIDUAL, column], places)

    novel = table("external_screening.csv").query(
        "`external set` == 'standardized external set' and cut == 'novel scaffold'").set_index("model")
    check("§4.7", "compounds in the novel-scaffold cut", 426,
          int(novel.loc[BEST_INDIVIDUAL, "compounds"]), 0)
    check("§4.7", "positives in the novel-scaffold cut", 14,
          int(novel.loc[BEST_INDIVIDUAL, "positives"]), 0)

    frequency = table("greedy_selection_frequency_by_criterion_standardized.csv")
    frequency = frequency.set_index(frequency.columns[0])
    for model in ("MolFormer-XL", "GIN"):
        check("§4.5", f"selection frequency of {model} in Greedy (AP)", 1.0,
              frequency.loc[model, "average precision"], 2)


def main():
    for section in (section_datasets, section_splits, section_tables, section_bootstrap,
                    section_threshold, section_optimism, section_collapse_ablation_external):
        section()
    report = pd.DataFrame(checks)
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.max_colwidth", 64):
        print(report.to_string(index=False))
    checked = report[report["verdict"] != "—"]
    mismatched = checked[checked["verdict"] == "MISMATCH"]
    print(f"\n{len(checked) - len(mismatched)} of {len(checked)} checked numbers match; "
          f"{len(report) - len(checked)} values printed without a target in the text.")
    if len(mismatched):
        print("\nMismatches:")
        print(mismatched.to_string(index=False))
    return 1 if len(mismatched) else 0


if __name__ == "__main__":
    sys.exit(main())
