"""Builds the benchmark notebook, the one that reproduces every number in the paper.

The notebook covers the whole study: sections 1 to 12 are the benchmark itself, and sections 13
to 16 are the analyses that close five of the eight limitations listed in the paper (bootstrap
confidence intervals, stacking calibration, threshold-free selection, the InChIKey collapse and
the label-conflict ablation). Nothing is refitted between the two parts.

Sections 7, 8 and 11 aggregate across models, and the result depends on which models enter the
average: the paper quotes the 22-model set (18 representations + 4 ensembles), while the full
set has 25. Both are computed and saved side by side, labelled by set.

Helper functions live in analysis_utils.py, which imports protocol_utils and core_utils.

Usage:
    python utils/logistic_regression/build_benchmark_notebook.py
"""

import json
import os

NOTEBOOK = "benchmark_four_protocols.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.strip("\n")})


# =========================================================================================
md("""
# Benchmark notebook — frozen encoders and ensembles for geroprotector screening

Runnable record of the experiments reported in the accompanying manuscript (under review). The
prose here documents the code and is written for this repository; it is not an excerpt of the
manuscript, and the manuscript is not included.

**What runs here.** Each of 18 frozen encoders feeds one L2 logistic regression. Seven
ensembles are built on top of them. Every model is scored in four settings, crossing two
datasets (merged by standardized InChIKey, or deduplicated by canonical SMILES only) with two
splitting rules (grouped by Bemis–Murcko skeleton, or stratified at random), each repeated over
5 folds × 3 seeds.

Sections 1 to 12 run the benchmark. Sections 13 to 16 add the uncertainty, calibration,
duplicate-counting and label-policy analyses. Nothing is refitted between the two parts: the
same encoders, folds and 1,080 member checkpoints are reused throughout.

**Two model sets, and why both are reported.** Sections 7, 8 and 11 aggregate *across models*: the mean Δ per model and the Spearman correlation between rankings. The result depends on which models enter the average, and this study grew by three ensembles along the way — Greedy (AP), Platt-calibrated stacking and isotonic-calibrated stacking. The paper quotes the **22-model** set (18 representations + 4 ensembles): Δ F1 of +0.042 and +0.0215, Δ ROC-AUC of +0.041 and +0.024, the −0.016 drop with 19 of 22 models declining, and the minimum Spearman of 0.948. Over all 25 models the same quantities are +0.047, +0.024, +0.042, +0.025, −0.021 with 22 of 25 declining, and a minimum Spearman of 0.862. Both are computed, printed side by side and saved to `across_model_aggregates_by_set.csv`. Everything else — per-model metrics, ensembles, confidence intervals, the external set — is model by model and does not depend on the choice.

| Section | What it adds |
|---|---|
| 13 | Confidence intervals and paired tests on the reference setting |
| 14 | Stacking calibration and a threshold-free member-selection criterion |
| 15 | One prediction per substance, to isolate duplicate counting |
| 16 | The three label policies for the 82 conflicting substances |

Three things stay open and are not computable here: MAT and R-MAT need a package this
environment cannot install, the external set has been read before, and a frozen extractor with
a linear head is a design choice rather than a defect.

The pipeline:

1. **Chemical standardization** of the training set, merging structural duplicates.
2. **Embeddings** from 18 encoders, checkpointed on disk. MAT and R-MAT are excluded because the package that produces them (`huggingmolecules`) is not available here. The unstandardized dataset reuses the embeddings computed for it earlier.
3. **L2 logistic regression** per representation, with `C` chosen by nested inner CV, over 5 folds × 3 seeds (42, 43, 44) = 15 folds. Metrics at threshold 0.5, on train and test.
4. **Ensembles** over the 18 representations: soft voting, ROC-AUC-weighted soft voting, and stacking.
5. **Ensemble evaluation** on the same 15 folds.
6. **Greedy selection**, decided inside each outer training fold only, compared against the best individual representation.
7. **Ranking** by F1.
8. **Scaffold vs random** in each dataset.
9. **Diversity** of the greedy ensemble (heat maps).
10. **Threshold sweep** from 0.40 to 0.60.
11. **Standardized vs unstandardized dataset.**
12. **External validation:** the best individual and the best ensemble of the scaffold split, refit on the full training set and read **once** on the external set (Precision@K, Recall@K, BEDROC), in both the standardized and the unstandardized version.
13. **Scaffold bootstrap and the Nadeau–Bengio test** on the standardized dataset (limitation i).
14. **Stacking calibration and threshold-free selection** (limitations vi and v).
15. **InChIKey collapse** of the unstandardized predictions (limitation iii).
16. **Label-conflict ablation** (limitation iv).

Steps 3 to 8 run on both datasets and under both splits: **scaffold** (Bemis–Murcko skeletons never appear in train and test at the same time) and **random stratified**. The greedy, diversity and threshold analyses (6, 9 and 10) use the scaffold split only, and the best individual representation is the one with the highest mean test F1 under the scaffold split.
""")

md("""
## Setup
""")
code("""
import os
import sys
import json
import importlib
import subprocess

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

# ---- paths: this notebook runs from notebooks/ -------------------------------------------
ROOT = os.path.abspath("..")                            # the repository root
sys.path.insert(0, os.path.join(ROOT, "utils", "logistic_regression"))
import analysis_utils as U
U = importlib.reload(U)   # a reused kernel will not pick up module edits without this

UNSTANDARDIZED_CSV = os.path.join(ROOT, "data", "gerollm_reconstructed", "primary_dataset_dedup.csv")
STANDARDIZED_DIR = os.path.join(ROOT, "data", "gerollm_standardized")
STANDARDIZED_CSV = os.path.join(STANDARDIZED_DIR, "primary_standardized.csv")
CHEMISTRY_UNSTANDARDIZED_CSV = os.path.join(STANDARDIZED_DIR, "standardization_audit_unstandardized_rows.csv")
CHEMISTRY_STANDARDIZED_CSV = os.path.join(STANDARDIZED_DIR, "standardization_audit_standardized.csv")

# Embedding folders: one .npy per encoder, per dataset.
EMBEDDINGS_DIR = {
    "standardized": os.path.join(ROOT, "data", "embeddings", "standardized_train"),
    "unstandardized": os.path.join(ROOT, "data", "embeddings", "unstandardized_train"),
}
EMBEDDINGS_SCRIPT = os.path.join(ROOT, "utils", "embeddings", "generate_embeddings.py")
# The encoders run in the `embeddings` conda environment (torch, transformers,
# torch_geometric, gensim, selfies, unimol_tools); this notebook runs in the analysis one.
EMBEDDINGS_PYTHON = os.path.expanduser("~/miniconda3/envs/embeddings/bin/python")

RESULTS_DIR = os.path.join(ROOT, "results")
checkpoints = U.Checkpoints(os.path.join(RESULTS_DIR, "checkpoints", "member_fits"))
FINAL_MODELS = os.path.join(RESULTS_DIR, "models", "final_models_standardized.joblib")


def out(name):
    folder = os.path.join(RESULTS_DIR, {".csv": "tables", ".png": "figures"}[os.path.splitext(name)[1]])
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


# ---- protocol ----------------------------------------------------------------------------
SEEDS = (42, 43, 44)
N_SPLITS = 5
C_GRID = np.logspace(-4, 2, 7)          # 1e-4 .. 1e2, seven values
DATASETS = ["standardized", "unstandardized"]
DATASET_NAME = {"standardized": "standardized dataset", "unstandardized": "unstandardized dataset"}
PROTOCOLS = ["scaffold", "random"]
PROTOCOL_NAME = {"scaffold": "scaffold split", "random": "random stratified split"}
SETTINGS = [(dataset, protocol) for dataset in DATASETS for protocol in PROTOCOLS]
N_JOBS = 5                               # inner folds in parallel

ENCODERS = {  # name in the study -> key of the embedding file
    "MolFormer-XL": "molformer", "SELFormer": "selformer", "ChemBERTa-2": "chembert",
    "ChemBERTa-3": "chemberta3", "MolDeBERTa": "moldeberta", "MolCLR": "molclr",
    "Mole-BERT": "molebert", "GIN": "gin", "GraphMVP": "graphmvp", "Uni-Mol": "unimol",
    "ECFP4-2048": "ecfp4", "Atom-Pair": "atompair", "Graph-FP": "graphfp", "TT": "tt",
    "Mol2Vec": "mol2vec", "ChemGPT": "chemgpt", "ChemFM": "chemfm", "CLAMP": "clamp",
}
# MAT and R-MAT are left out: they depend on the huggingmolecules package, which only exists
# on GitHub and is not installed in the `embeddings` environment. To include them, install the
# package and add "MAT": "mat" and "R-MAT": "rmat" above.
BINARY_FPS = {"ECFP4-2048", "Atom-Pair", "Graph-FP", "TT"}


def embedding_files(dataset, chave):
    \"\"\"Paths of the embedding matrix and of the labels of one encoder in one dataset.\"\"\"
    folder = EMBEDDINGS_DIR[dataset]
    if dataset == "unstandardized" and chave == "ecfp4":      # this cut stores ECFP4 under another name
        return os.path.join(folder, "fingerprints.npy"), os.path.join(folder, "labels_fingerprints.npy")
    return os.path.join(folder, f"Embeddings_{chave}.npy"), os.path.join(folder, f"labels_{chave}.npy")
""")

# ---------------------------------------------------------------------------- 1
md("""
## 1. Dataset and chemical standardization

Every SMILES goes through RDKit standardization: cleanup, removal of salts and solvents (fragment parent), uncharging, and canonical tautomer. Rows sharing a standardized InChIKey are the same substance and are merged into **a single row**:

- the SMILES becomes the standardized canonical tautomer;
- the label is positive if any form is positive, the same rule the dataset already uses for conflicts between sources.

The unstandardized dataset is the conventional one, with its structural duplicates. The standardization result is cached as a CSV; re-running loads the file instead of recomputing it.
""")
code("""
os.makedirs(STANDARDIZED_DIR, exist_ok=True)
unstandardized = pd.read_csv(UNSTANDARDIZED_CSV)
if os.path.exists(STANDARDIZED_CSV):
    standardized = pd.read_csv(STANDARDIZED_CSV)
    chemistry_unstandardized = pd.read_csv(CHEMISTRY_UNSTANDARDIZED_CSV)
    print(f"[checkpoint] standardized dataset loaded from {os.path.basename(STANDARDIZED_CSV)}")
else:
    standardized, chemistry_unstandardized, _ = U.standardize_dataset(unstandardized)
    standardized.to_csv(STANDARDIZED_CSV, index=False)
    chemistry_unstandardized.to_csv(CHEMISTRY_UNSTANDARDIZED_CSV, index=False)

summary = {
    "rows in the unstandardized dataset": len(unstandardized),
    "distinct substances (standardized dataset)": len(standardized),
    "rows merged as duplicates": len(unstandardized) - len(standardized),
    "substances that merged more than one row": int((standardized["n_forms"] > 1).sum()),
    "substances with conflicting labels (resolved positive)": int(standardized["conflicting_label"].sum()),
    "positives before / after": f"{int(unstandardized['label'].sum())} / {int(standardized['label'].sum())}",
    "prevalence after": f"{standardized['label'].mean():.2%}",
}
display(pd.Series(summary, name="value").to_frame())
""")
code("""
# How many duplicates each standardization step revealed
display(U.duplicate_audit(chemistry_unstandardized))
examples = standardized.query("n_forms > 1").head(8)[["smiles", "smiles_original", "label", "n_forms",
                                                        "conflicting_label", "name"]]
display(examples)
""")
code("""
# Scaffold groups, used by the scaffold split. In the unstandardized dataset they are
# computed row by row, before any merging.
if os.path.exists(CHEMISTRY_STANDARDIZED_CSV):
    chemistry = pd.read_csv(CHEMISTRY_STANDARDIZED_CSV)
else:
    chemistry = U.chemistry_table(standardized["smiles"].tolist())
    chemistry.to_csv(CHEMISTRY_STANDARDIZED_CSV, index=False)
assert len(chemistry) == len(standardized) and len(chemistry_unstandardized) == len(unstandardized)
repeated = int(chemistry["L5_inchikey"].duplicated().sum())
if repeated:
    print(f"Warning: {repeated} compounds still collide when the standardized SMILES is re-standardized.")

Y = {"standardized": standardized["label"].astype(int).to_numpy(),
     "unstandardized": unstandardized["label"].astype(int).to_numpy()}
GROUPS = {"standardized": chemistry["group"].to_numpy(),
          "unstandardized": chemistry_unstandardized["group"].to_numpy()}
for dataset in DATASETS:
    print(f"{DATASET_NAME[dataset]}: {len(Y[dataset])} compounds, {Y[dataset].sum()} positives "
          f"({Y[dataset].mean():.2%}), {len(np.unique(GROUPS[dataset]))} scaffold groups")
""")

# ---------------------------------------------------------------------------- 2
md("""
## 2. Embeddings

One `.npy` file per encoder in the embeddings folder of each dataset. When this cell runs:

- if the folder is empty or incomplete, the missing embeddings are generated;
- if it is complete, the notebook **asks** whether to regenerate them (`y`) or reuse them (Enter).

To run unattended (with `nbconvert`, for instance), set `REGENERATE_EMBEDDINGS` to `True` or `False`, or the environment variable `REGENERATE_EMBEDDINGS=1`/`0`.

Generation runs in the `embeddings` conda environment through a separate script
(`utils/embeddings/generate_embeddings.py`), with the same functions that produced the
embeddings in the first place. Each encoder is a checkpoint, so an interrupted run resumes.
""")
code("""
REGENERATE_EMBEDDINGS = None   # None = ask; True = regenerate everything; False = reuse

if os.environ.get("REGENERATE_EMBEDDINGS") in ("0", "1"):
    REGENERATE_EMBEDDINGS = os.environ["REGENERATE_EMBEDDINGS"] == "1"


def embedding_ready(dataset, chave):
    file_path = embedding_files(dataset, chave)[0]
    return os.path.exists(file_path) and np.load(file_path, mmap_mode="r").shape[0] == len(Y[dataset])


ready = [n for n, chave in ENCODERS.items() if embedding_ready("standardized", chave)]
missing = [n for n in ENCODERS if n not in ready]
print(f"Folder: {EMBEDDINGS_DIR['standardized']}")
print(f"{len(ready)} of {len(ENCODERS)} encoders have complete embeddings for the standardized dataset.")

if REGENERATE_EMBEDDINGS is None and ready:
    resposta = input(f"Regenerate the embeddings? {len(ready)} ready, "
                     f"{len(missing)} missing. [y = regenerate all / Enter = reuse] ")
    REGENERATE_EMBEDDINGS = resposta.strip().lower() in ("y", "yes")

to_generate = list(ENCODERS) if REGENERATE_EMBEDDINGS else missing
if to_generate:
    command = [EMBEDDINGS_PYTHON, EMBEDDINGS_SCRIPT, "--csv", STANDARDIZED_CSV,
               "--coluna", "smiles", "--set_label", "label", "--saida", EMBEDDINGS_DIR["standardized"],
               "--encoders", *to_generate] + (["--force"] if REGENERATE_EMBEDDINGS else [])
    print("Generating:", ", ".join(to_generate))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for row in process.stdout:
        if row.startswith(("[ok]", "[error]", "[checkpoint]", "Summary")):
            print(row, end="")
    process.wait()
else:
    print("Reusing the embeddings already on disk.")
""")
code("""
# Only encoders with embeddings in both datasets take part, so that the comparison uses the
# same representations throughout.
NAMES = [n for n, chave in ENCODERS.items() if all(embedding_ready(b, chave) for b in DATASETS)]
absent = [n for n in ENCODERS if n not in NAMES]
if absent:
    print(f"WARNING: {len(absent)} encoder(s) without embeddings, dropped from what follows: {absent}")

X = {dataset: {} for dataset in DATASETS}
for dataset in DATASETS:
    for n in NAMES:
        file_path, labels_file = embedding_files(dataset, ENCODERS[n])
        X[dataset][n] = np.load(file_path).astype(np.float32)
        labels = np.load(labels_file).astype(int)
        assert X[dataset][n].shape[0] == len(Y[dataset]) and (labels == Y[dataset]).all(), (dataset, n)
display(pd.DataFrame({DATASET_NAME[b]: {n: X[b][n].shape[1] for n in NAMES} for b in DATASETS}).T)
""")

# ---------------------------------------------------------------------------- 3
md("""
## 3. L2 logistic regression per representation

- **Model:** logistic regression with an L2 penalty, `class_weight="balanced"` and standardized features; for the four binary fingerprints, bits present in fewer than 1% of the training molecules are dropped first.
- **Choosing C:** inside each outer training fold, an inner 5-fold CV selects `C` over the grid 10⁻⁴ to 10² by ROC-AUC.
- **Validation:** 5 folds × 3 seeds = 15 outer folds, under both splits and on both datasets.
- **Metrics:** accuracy, recall, precision, F1, balanced accuracy and MCC at threshold 0.5, plus ROC-AUC and average precision (the area under the precision-recall curve), which are threshold-free, on train and test.

Before any fit, an explicit audit of the scaffold split: for every fold, how many scaffold groups appear in train and test at the same time. Under the scaffold split that count must be zero and the notebook stops if it is not; under the random split it shows how many skeletons the test fold shares with training.

Every fit is checkpointed.
""")
code("""
folds = {}
for dataset in DATASETS:
    folds[(dataset, "random")] = U.repeated_random_folds(Y[dataset], n_splits=N_SPLITS, seeds=SEEDS)
    folds[(dataset, "scaffold")] = U.repeated_scaffold_folds(Y[dataset], GROUPS[dataset], n_splits=N_SPLITS,
                                                          seeds=SEEDS)
indices = {c: {(s, k): (tr, te) for s, k, tr, te in folds[c]} for c in SETTINGS}

# Scaffold leakage audit: under the scaffold split no skeleton may be in train and test at once
audit = pd.concat([U.shared_scaffolds(folds[(b, p)], GROUPS[b]).assign(
    dataset=DATASET_NAME[b], protocol=PROTOCOL_NAME[p]) for b, p in SETTINGS], ignore_index=True)
audit.to_csv(out("ex033_verificacao_scaffold.csv"), index=False)
summary = audit.groupby(["dataset", "protocol"])[["scaffolds in test", "scaffolds in train and test"]].agg(["mean", "std", "max"])
print("Scaffolds shared between train and test, per fold (15 folds)")
display(summary.round(2))
under_scaffold = audit[audit["protocol"] == PROTOCOL_NAME["scaffold"]]
assert under_scaffold["scaffolds in train and test"].eq(0).all(), "scaffold leakage"
print("OK: under the scaffold split, no skeleton appears in train and test at the same time.")

members = {}
for dataset, protocol in SETTINGS:
    members[(dataset, protocol)] = U.fit_members(
        X[dataset], Y[dataset], folds[(dataset, protocol)], NAMES, BINARY_FPS, C_GRID, checkpoints, protocol,
        groups=GROUPS[dataset] if protocol == "scaffold" else None, n_jobs=N_JOBS,
        prefix=f"{dataset}_{protocol}")
    print(f"{DATASET_NAME[dataset]}, {protocol}: {len(members[(dataset, protocol)])} fits "
          f"(15 folds x {len(NAMES)} representations)")

individual_metrics = pd.concat(
    [U.metrics_table(Y[b], indices[(b, p)], members[(b, p)], p, b) for b, p in SETTINGS],
    ignore_index=True)
individual_metrics.to_csv(out("ex033_metricas_individuais_por_fold.csv"), index=False)
""")
code("""
def show_per_fold(table, model):
    \"\"\"Per-fold metrics of one model, train and test, plus the mean and sd over the 15 folds.\"\"\"
    for dataset, protocol in SETTINGS:
        part = table.query("dataset == @dataset and protocol == @protocol and model == @model")
        wide = part.pivot_table(index=["seed", "fold"], columns="part", values=U.METRICS)
        wide.columns = [f"{U.METRIC_NAMES[m]} ({p})" for m, p in wide.columns]
        wide = wide[[f"{U.METRIC_NAMES[m]} ({p})" for m in U.METRICS for p in ("train", "test")]]
        mean, sd = wide.mean(), wide.std()
        wide.loc[("mean", ""), :] = mean
        wide.loc[("standard deviation", ""), :] = sd
        print(f"{model} — {DATASET_NAME[dataset]} — {PROTOCOL_NAME[protocol]}")
        display(wide.round(4))


for name in NAMES:
    show_per_fold(individual_metrics, name)
""")
code("""
individual_means = U.mean_by_model(individual_metrics)
individual_means.to_csv(out("ex033_metricas_individuais_media.csv"), index=False)
for dataset, protocol in SETTINGS:
    for part in ("test", "train"):
        selected = (individual_means.query("dataset == @dataset and protocol == @protocol and part == @part")
                   .set_index("model").sort_values("f1_mean", ascending=False))
        table = pd.DataFrame({U.METRIC_NAMES[m]: U.mean_sd(selected[f"{m}_mean"], selected[f"{m}_std"])
                               for m in U.METRICS})
        print(f"Mean ± sd over the 15 folds — {DATASET_NAME[dataset]} — {PROTOCOL_NAME[protocol]} — {part}")
        display(table)
""")
md("""
### 3.1 Final models on the full dataset

Each representation is refit on the whole standardized dataset; the unstandardized dataset only serves as a comparison and gets its final models in section 12. The `C` values and the out-of-fold probabilities that feed the final ensembles (weights, meta-model and greedy subset) come from a 5-fold scaffold CV over the full dataset. The models are saved for use in the ensembles and in later predictions.
""")
code("""
if os.path.exists(FINAL_MODELS):
    final_models = joblib.load(FINAL_MODELS)
    print(f"[checkpoint] final models loaded from {os.path.basename(FINAL_MODELS)}")
else:
    final_models = U.fit_final_models(X["standardized"], Y["standardized"], NAMES, BINARY_FPS, C_GRID,
                              GROUPS["standardized"], n_jobs=N_JOBS)
    os.makedirs(os.path.dirname(FINAL_MODELS), exist_ok=True)
    joblib.dump(final_models, FINAL_MODELS)
display(pd.DataFrame({"C": final_models["C"], "weight in the weighted vote": final_models["roc_auc_weights"]}).round(4))
print("Members of the final greedy ensemble:", final_models["greedy_members"])
""")

# ---------------------------------------------------------------------------- 4
md("""
## 4. Ensembles over the 18 representations

1. **Soft voting:** the mean of the member probabilities.
2. **Weighted soft voting (ROC-AUC):** that mean weighted by each member's inner-CV ROC-AUC, measured on the training part only, with weights summing to 1.
3. **Stacking:** a meta-model (L2 logistic regression, `C=1`, `class_weight="balanced"`) fitted on the members' inner out-of-fold probabilities, that is, on predictions for compounds each member has not seen.

Three more ensembles complete the set:

4. **Stacking (isotonic)** and 5. **Stacking (Platt):** the same meta-model with a calibrator fitted **inside the fold**, on the meta-model's own inner out-of-fold scores. `class_weight="balanced"` shifts the stacking probabilities upward, so at threshold 0.5 it predicts positive too often: it has the highest ROC-AUC among the ensembles and the lowest F1. Calibration is a monotone map, so **ROC-AUC and average precision do not change** — what changes is where 0.5 falls. Isotonic regression imposes no functional form; Platt scaling fits a one-parameter sigmoid and is steadier with few positives.
6. **Greedy (AP):** the same greedy search, maximizing the average precision of the averaged probabilities instead of F1 at threshold 0.5. The F1 variant leaves the inner score optimized exactly at 0.5, which makes the threshold study circular for that ensemble (section 10). Average precision needs no threshold and is the ranking metric closest to screening use at 17% prevalence.

In all of them the weights, the meta-model, the calibrator and the subset are decided from the training part of each fold alone.
""")

# ---------------------------------------------------------------------------- 5
md("""
## 5. Ensemble evaluation on the 15 folds

The same 15 folds and the same metrics as the individual models, at threshold 0.5, on train and test, in both datasets and under both splits. Greedy selection (section 6) is computed here as well.
""")
code("""
ensembles = {}
for dataset, protocol in SETTINGS:
    setting = (dataset, protocol)
    ensembles[setting] = {}
    for (seed, k), (tr, te) in indices[setting].items():
        records = {n: members[setting][(seed, k, n)] for n in NAMES}
        for name, record in U.fold_ensembles(Y[dataset], tr, records, NAMES, seed).items():
            ensembles[setting][(seed, k, name)] = record

ensemble_metrics = pd.concat(
    [U.metrics_table(Y[b], indices[(b, p)], ensembles[(b, p)], p, b) for b, p in SETTINGS],
    ignore_index=True)
ensemble_metrics.to_csv(out("ex033_metricas_ensembles_por_fold.csv"), index=False)

for name in U.ALL_ENSEMBLES:
    show_per_fold(ensemble_metrics, name)
""")
code("""
ensemble_means = U.mean_by_model(ensemble_metrics)
ensemble_means.to_csv(out("ex033_metricas_ensembles_media.csv"), index=False)
for dataset, protocol in SETTINGS:
    selected = ensemble_means.query("dataset == @dataset and protocol == @protocol").set_index(["part", "model"])
    table = pd.DataFrame({f"{U.METRIC_NAMES[m]} ({p})": U.mean_sd(selected.loc[p, f"{m}_mean"],
                                                                     selected.loc[p, f"{m}_std"])
                           for m in U.METRICS for p in ("test", "train")})
    print(f"Mean ± sd over the 15 folds — {DATASET_NAME[dataset]} — {PROTOCOL_NAME[protocol]}")
    display(table)
""")

# ---------------------------------------------------------------------------- 6
md("""
## 6. Greedy selection (scaffold split)

Inside each fold, the search adds one representation at a time to the soft-voting average, picking the one that most increases F1 (at threshold 0.5) on the inner out-of-fold probabilities of the training part, and keeps the best step along the path. The test fold never informs the choice.

The analysis uses the scaffold split. The reference is the **best individual representation under the scaffold split** (highest mean test F1 over the 15 folds) in each dataset:

- the table compares the greedy ensemble against that model on the test folds;
- in the path plot, the dashed line is that model's mean inner F1, measured on the same inner out-of-fold probabilities that guide the search.
""")
code("""
best_individual = {}
for dataset in DATASETS:
    test = individual_means.query("dataset == @dataset and protocol == 'scaffold' and part == 'test'")
    best_individual[dataset] = test.set_index("model")["f1_mean"].idxmax()
    print(f"Best individual representation, scaffold split — {DATASET_NAME[dataset]}: {best_individual[dataset]}")

frequencies, consensus = {}, {}
for dataset in DATASETS:
    setting = (dataset, "scaffold")
    frequencies[dataset] = U.selection_frequency(ensembles[setting], NAMES)
    sizes = [len(r["members"]) for (s, k, m), r in ensembles[setting].items() if m == "Greedy (F1)"]
    print(f"{DATASET_NAME[dataset]}: {len(sizes)} folds, median subset size {np.median(sizes):.0f} "
          f"(from {min(sizes)} to {max(sizes)} members)")
    consensus[dataset] = frequencies[dataset].query("frequency >= 0.5").index.tolist()
table = pd.concat({DATASET_NAME[b]: frequencies[b]["frequency"] for b in DATASETS}, axis=1)
table.to_csv(out("ex033_frequencia_gulosa_scaffold.csv"))
display(table.sort_values(DATASET_NAME["standardized"], ascending=False).round(3))
for dataset in DATASETS:
    print(f"Consensus members ({DATASET_NAME[dataset]}, selected in >= 50% of the folds): {consensus[dataset]}")
""")
code("""
rows = []
for dataset in DATASETS:
    for model, tipo in ((best_individual[dataset], "best individual"), ("Greedy (F1)", "greedy ensemble")):
        source = ensemble_means if model == "Greedy (F1)" else individual_means
        row = source.query("dataset == @dataset and protocol == 'scaffold' and part == 'test' "
                            "and model == @model").iloc[0]
        rows.append({"dataset": DATASET_NAME[dataset], "kind": tipo, "model": model,
                       **{U.METRIC_NAMES[m]: f"{row[f'{m}_mean']:.4f} ± {row[f'{m}_std']:.4f}"
                          for m in U.METRICS}})
greedy_vs_individual = pd.DataFrame(rows)
greedy_vs_individual.to_csv(out("ex033_guloso_x_melhor_individual_scaffold.csv"), index=False)
print("Test folds, scaffold split (mean ± sd over the 15 folds)")
display(greedy_vs_individual)
""")
code("""
fig, axes = plt.subplots(1, len(DATASETS), figsize=(7 * len(DATASETS), 4), sharey=True)
for ax, dataset in zip(axes, DATASETS):
    setting = (dataset, "scaffold")
    for (seed, k, name), record in ensembles[setting].items():
        if name == "Greedy (F1)":
            path = record["path"]
            ax.plot(path["n_members"], path["f1_inner"], color="tab:blue", alpha=0.35)
    best_inner_f1 = np.mean([
        f1_score(Y[dataset][tr], (members[setting][(s, k, best_individual[dataset])]["inner_oof"]
                               >= U.DEFAULT_THRESHOLD).astype(int), zero_division=0)
        for (s, k), (tr, te) in indices[setting].items()])
    ax.axhline(best_inner_f1, color="tab:red", ls="--", lw=1.5,
               label=f"best individual ({best_individual[dataset]})")
    ax.set_title(f"Greedy selection path — {DATASET_NAME[dataset]}, scaffold")
    ax.set_xlabel("number of members")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
axes[0].set_ylabel("inner F1 (training part)")
fig.tight_layout()
fig.savefig(out("greedy_path_scaffold.png"), dpi=150, bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------------------- 7
md("""
## 7. Ranking

Individual models, the three all-representation ensembles and the two greedy ensembles, ordered by mean test F1 over the 15 folds (threshold 0.5), in each dataset and under each split.

The **rank** column is among all 25 models; **rank (paper's 22)** is among the 22 the paper quotes. They differ for anything below one of the three later ensembles: Graph-FP, for instance, is 15th among the 22 and 16th among the 25.
""")
code("""
all_means = pd.concat([individual_means, ensemble_means], ignore_index=True)

# Model sets for the across-model aggregations (sections 7, 8 and 11). The paper quotes the
# 22-model numbers; the 25-model set adds the three later ensembles. Both are computed in every
# aggregation, and the set is named in the output.
ALL_MODELS = sorted(all_means["model"].unique())
PAPER_MODELS = [m for m in ALL_MODELS if m not in U.LATER_ENSEMBLES]
MODEL_SETS = {"22 (paper)": PAPER_MODELS, "25 (full)": ALL_MODELS}
aggregates = []        # filled in sections 8 and 11, saved at the end of section 11
for set_label, models in MODEL_SETS.items():
    print(f"model set {set_label}: {len(models)} models")
assert len(PAPER_MODELS) == 22 and len(ALL_MODELS) == 25, "unexpected model set"

rankings = {}
for dataset, protocol in SETTINGS:
    test = all_means.query("dataset == @dataset and protocol == @protocol and part == 'test'").set_index("model")
    ranking = pd.DataFrame({U.METRIC_NAMES[m]: U.mean_sd(test[f"{m}_mean"], test[f"{m}_std"])
                            for m in U.METRICS})
    ranking["kind"] = ["ensemble" if m in U.ALL_ENSEMBLES else "individual" for m in ranking.index]
    ranking = ranking.loc[test["f1_mean"].sort_values(ascending=False).index]
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    # Rank within the paper's 22 models: that is the one the text cites (Graph-FP, for
    # instance, is 15th among the 22 and 16th among the 25).
    in_paper_set = [m for m in ranking.index if m in PAPER_MODELS]
    ranking.insert(1, "rank (paper's 22)",
                   [in_paper_set.index(m) + 1 if m in PAPER_MODELS else None for m in ranking.index])
    rankings[(dataset, protocol)] = ranking
    ranking.to_csv(out(f"ex033_ranking_{dataset}_{protocol}.csv"))
    print(f"Ranking by mean test F1 — {DATASET_NAME[dataset]} — {PROTOCOL_NAME[protocol]} "
          f"(mean ± sd over the 15 folds)")
    display(ranking)
""")

# ---------------------------------------------------------------------------- 8
md("""
## 8. Scaffold vs random

Test F1 and ROC-AUC of every model under both splits (mean ± standard deviation over the 15 folds), in each dataset, and the difference random − scaffold. Under the random split, compounds sharing a skeleton fall on both sides, so the test fold looks more like training; the difference measures how much of the performance depends on having seen similar skeletons. The folds of the two splits are different, so the comparison is between means rather than fold by fold, and the standard deviation of Δ is √(sd_scaffold² + sd_random²). The mean Δ across models comes with the standard deviation across models.

The Spearman correlation between the F1 values of the two splits says whether the ordering of the models survives. The mean Δ and the Spearman are aggregations across models, so both model sets (22 and 25) are reported; the paper quotes the 22.
""")
code("""
protocol_comparison = {}
for dataset in DATASETS:
    table = U.compare(all_means.query("dataset == @dataset"), "protocol", "scaffold", "random")
    table = table.sort_values("F1 (scaffold)", ascending=False)
    table.insert(0, "kind", ["ensemble" if m in U.ALL_ENSEMBLES else "individual" for m in table.index])
    protocol_comparison[dataset] = table
    table.to_csv(out(f"scaffold_vs_random_{dataset}.csv"))
    print(f"{DATASET_NAME[dataset]} — test folds, mean ± sd over the 15 folds")
    for set_label, models in MODEL_SETS.items():
        sub = table.loc[[m for m in table.index if m in models]]
        rho = spearmanr(sub["F1 (scaffold)"], sub["F1 (aleatorio)"]).statistic
        delta_f1 = sub["Δ F1 (aleatorio − scaffold)"]
        delta_auc = sub["Δ ROC-AUC (aleatorio − scaffold)"]
        print(f"  [{len(sub)} models, set {set_label}] "
              f"mean Δ F1 across models (random − scaffold): {delta_f1.mean():+.4f} ± {delta_f1.std():.4f}; "
              f"mean Δ ROC-AUC: {delta_auc.mean():+.4f} ± {delta_auc.std():.4f}; "
              f"Spearman between the F1 values: {rho:.3f}")
        aggregates.append({"comparison": "random − scaffold", "held fixed": DATASET_NAME[dataset],
                          "model set": set_label, "models": len(sub),
                          "mean Δ F1": delta_f1.mean(), "Δ F1 sd across models": delta_f1.std(),
                          "mean Δ ROC-AUC": delta_auc.mean(),
                          "Δ ROC-AUC sd across models": delta_auc.std(),
                          "models with Δ F1 < 0": int((delta_f1 < 0).sum()), "Spearman": rho})
    display(U.format_sd(table))
""")
code("""
fig, axes = plt.subplots(1, len(DATASETS), figsize=(8 * len(DATASETS), 9), sharex=True)
for ax, dataset in zip(axes, DATASETS):
    test = all_means.query("dataset == @dataset and part == 'test'")
    order = protocol_comparison[dataset].index[::-1]
    positions = np.arange(len(order))
    for offset, protocol, color in ((-0.2, "scaffold", "tab:blue"), (0.2, "random", "tab:orange")):
        values_ = test.query("protocol == @protocol").set_index("model").loc[order]
        ax.barh(positions + offset, values_["f1_mean"], height=0.4, xerr=values_["f1_std"],
                color=color, label=PROTOCOL_NAME[protocol], error_kw={"lw": 0.8})
    ax.set_yticks(positions)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("test F1 (mean ± sd over the 15 folds)")
    ax.set_title(f"Scaffold vs random — {DATASET_NAME[dataset]}")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(out("scaffold_vs_random.png"), dpi=150, bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------------------- 9
md("""
## 9. Diversity of the greedy ensemble (scaffold split)

Pairwise measures between the consensus members of the greedy ensemble under the scaffold split (those selected in at least half of the folds), computed on the test predictions of the seed-42 repetition at threshold 0.5, where every compound is predicted exactly once by a model that has not seen it:

- **Double-fault:** the fraction of compounds both get wrong (lower = more complementary).
- **Q-statistic:** 0 = independent errors, 1 = coincident errors.
- **Disagreement:** the fraction of compounds the two disagree on (higher = more diverse).
- **Correlation coefficient:** the correlation between the two hit/miss vectors.
""")
code("""
for dataset in DATASETS:
    setting = (dataset, "scaffold")
    members_div = consensus[dataset]
    if len(members_div) < 2:
        print(f"{DATASET_NAME[dataset]}: fewer than 2 consensus members, no diversity map.")
        continue
    acertos = U.oof_hits(Y[dataset], indices[setting], members[setting], members_div, seed=SEEDS[0])
    fig, axes = plt.subplots(1, 4, figsize=(5 * 4, 4.4))
    for ax, (title, measure) in zip(axes, U.DIVERSITY_MEASURES.items()):
        matrix = U.pairwise_matrix(measure, acertos, members_div)
        image = ax.imshow(matrix, cmap="viridis")
        ax.set_xticks(range(len(members_div)))
        ax.set_xticklabels(members_div, rotation=90, fontsize=8)
        ax.set_yticks(range(len(members_div)))
        ax.set_yticklabels(members_div, fontsize=8)
        for i in range(len(members_div)):
            for j in range(len(members_div)):
                if i != j:
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(f"Diversity of the greedy ensemble — {DATASET_NAME[dataset]}, scaffold split")
    fig.tight_layout()
    fig.savefig(out(f"diversity_{dataset}_scaffold.png"), dpi=150, bbox_inches="tight")
    plt.show()
""")

# ---------------------------------------------------------------------------- 10
md("""
## 10. Threshold sweep (scaffold split)

Every metric so far used the threshold 0.5. Here the best individual model and the best ensemble **under the scaffold split** (highest mean test F1) are evaluated over a threshold grid from 0.40 to 0.60 in steps of 0.01, in each dataset:

- **Out-of-fold curve (solid line):** mean F1 over the 15 folds on the inner out-of-fold probabilities of the training part, at each threshold.
- **Test curve (dashed line):** mean test F1 over the 15 folds at each threshold. Shown for reference only; it never informs the choice.
- **Selected threshold:** one per model, at the peak of the out-of-fold curve.
- **Circles:** both curves at 0.5. **Stars:** both curves at the selected threshold.

The table holds exactly the values of the circles and the stars, with the standard deviation over the 15 folds. The folds are the same at both thresholds, so the standard deviation of Δ is that of the per-fold difference. The test fold never enters the choice, but the out-of-fold scores of one fold come from a training set that contains test compounds of other folds; since the threshold is a single scalar, that leakage is mild.
""")
code("""
threshold_rows = []
fig, axes = plt.subplots(1, len(DATASETS), figsize=(8 * len(DATASETS), 4.5), sharey=True)
for ax, dataset in zip(axes, DATASETS):
    setting = (dataset, "scaffold")
    test = all_means.query("dataset == @dataset and protocol == 'scaffold' and part == 'test'")
    test = test.set_index("model")["f1_mean"]
    best_ensemble = test[[m for m in U.ALL_ENSEMBLES if m in test.index]].idxmax()

    for model, color in ((best_individual[dataset], "tab:blue"), (best_ensemble, "tab:red")):
        source = ensembles[setting] if model in U.ALL_ENSEMBLES else {
            (s, k, n): {**r, "inner_score": r["inner_oof"]} for (s, k, n), r in members[setting].items()}
        probas = {(s, k): r for (s, k, n), r in source.items() if n == model}
        per_fold, curve, threshold = U.threshold_sweep(Y[dataset], indices[setting], probas)
        ax.plot(curve.index, curve["f1_oof_mean"], ls="-", color=color, label=f"{model} — out-of-fold")
        ax.plot(curve.index, curve["f1_test_mean"], ls="--", color=color,
                label=f"{model} — test (reference only)")
        ax.scatter([0.5, 0.5], curve.loc[0.5, ["f1_oof_mean", "f1_test_mean"]], s=70,
                   facecolors="none", edgecolors=color, linewidths=2, zorder=3)
        ax.scatter([threshold, threshold], curve.loc[threshold, ["f1_oof_mean", "f1_test_mean"]], s=160,
                   marker="*", color=color, edgecolors="black", linewidths=0.6, zorder=4)
        by_threshold = per_fold.set_index(["threshold", "seed", "fold"]).sort_index()
        row = {"dataset": DATASET_NAME[dataset], "model": model, "selected threshold (out-of-fold peak)": threshold}
        for coluna, name in (("f1_oof", "out-of-fold F1"), ("f1_test", "test F1")):
            delta = by_threshold.loc[threshold, coluna] - by_threshold.loc[0.5, coluna]
            row.update({
                f"{name} at 0.5": curve.loc[0.5, f"{coluna}_mean"],
                f"{name} at 0.5 sd": curve.loc[0.5, f"{coluna}_std"],
                f"{name} at the selected threshold": curve.loc[threshold, f"{coluna}_mean"],
                f"{name} at the selected threshold sd": curve.loc[threshold, f"{coluna}_std"],
                f"Δ {name}": delta.mean(), f"Δ {name} sd": delta.std()})
        threshold_rows.append(row)
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.scatter([], [], s=70, facecolors="none", edgecolors="gray", linewidths=2,
               label="threshold 0.5")
    ax.scatter([], [], s=160, marker="*", color="gray", edgecolors="black", linewidths=0.6,
               label="selected threshold (out-of-fold peak)")
    ax.set_xlabel("threshold")
    ax.set_title(f"F1 as a function of the threshold — {DATASET_NAME[dataset]}, scaffold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("mean F1 (15 folds)")
fig.tight_layout()
fig.savefig(out("threshold_sweep_scaffold.png"), dpi=150, bbox_inches="tight")
plt.show()

threshold_table = pd.DataFrame(threshold_rows)
threshold_table.to_csv(out("threshold_sweep_scaffold.csv"), index=False)
print("Mean ± sd over the 15 folds")
display(U.format_sd(threshold_table))
""")

# ---------------------------------------------------------------------------- 11
md("""
## 11. Standardized vs unstandardized dataset

First, the eight test metrics of every model in each dataset and under each split (four tables, mean ± standard deviation over the 15 folds, ordered by F1). Then test F1 and ROC-AUC of every model on both datasets, side by side, and the difference standardized − unstandardized. The two datasets do not hold the same compounds (the standardized one merges structural duplicates), so the comparison is between means rather than compound by compound, and the standard deviation of Δ is √(sd_standardized² + sd_unstandardized²). The mean Δ across models comes with the standard deviation across models.

As in section 8, the mean Δ across models and the Spearman are reported for both model sets, and the paper quotes the 22 — including how many models decline under standardization, which is 19 of 22 (and 22 of 25).

In the unstandardized dataset, forms of the same substance (salt and free dataset, charged and neutral forms, tautomers) are separate rows. Under the random split one form can land in training and another in test, and the model is then scored on a compound it has already seen. The leakage table counts those cases under both splits; under the scaffold split it almost never happens, because forms of the same substance nearly always share a skeleton and fall in the same group.
""")
code("""
for dataset, protocol in SETTINGS:
    print(f"Test folds — {DATASET_NAME[dataset]} — {PROTOCOL_NAME[protocol]} (mean ± sd over the 15 folds)")
    display(rankings[(dataset, protocol)])
""")
code("""
keys_array = chemistry_unstandardized["L5_inchikey"].to_numpy()
valid = pd.notna(keys_array)
leakage = []
for protocol in PROTOCOLS:
    for seed, k, tr, te in folds[("unstandardized", protocol)]:
        in_training = set(keys_array[tr][valid[tr]])
        leakage.append({"protocol": PROTOCOL_NAME[protocol], "seed": seed, "fold": k,
                          "test compounds whose substance is in training":
                              int(sum(c in in_training for c in keys_array[te][valid[te]])),
                          "test compounds": len(te)})
leakage = pd.DataFrame(leakage)
leakage["fraction"] = leakage["test compounds whose substance is in training"] / leakage["test compounds"]
columns_ = ["test compounds whose substance is in training", "test compounds", "fraction"]
summary = leakage.groupby("protocol")[columns_].agg(["mean", "std"])
print("Duplicate-substance leakage in the unstandardized dataset (mean ± sd over the 15 folds)")
display(pd.DataFrame({c: U.mean_sd(summary[(c, "mean")], summary[(c, "std")]) for c in columns_}))
""")
code("""
dataset_comparison = {}
for protocol in PROTOCOLS:
    table = U.compare(all_means.query("protocol == @protocol"), "dataset", "unstandardized", "standardized")
    table = table.sort_values("F1 (standardized)", ascending=False)
    table.insert(0, "kind", ["ensemble" if m in U.ALL_ENSEMBLES else "individual" for m in table.index])
    dataset_comparison[protocol] = table
    table.to_csv(out(f"standardized_vs_unstandardized_{protocol}.csv"))
    print(f"{PROTOCOL_NAME[protocol]} — test folds, mean ± sd over the 15 folds")
    for set_label, models in MODEL_SETS.items():
        sub = table.loc[[m for m in table.index if m in models]]
        rho = spearmanr(sub["F1 (unstandardized)"], sub["F1 (standardized)"]).statistic
        delta_f1 = sub["Δ F1 (standardized − unstandardized)"]
        delta_auc = sub["Δ ROC-AUC (standardized − unstandardized)"]
        print(f"  [{len(sub)} models, set {set_label}] "
              f"mean Δ F1 across models (standardized − unstandardized): {delta_f1.mean():+.4f} ± {delta_f1.std():.4f}; "
              f"mean Δ ROC-AUC: {delta_auc.mean():+.4f} ± {delta_auc.std():.4f}; "
              f"Spearman between the F1 values: {rho:.3f}; "
              f"models that decline: {int((delta_f1 < 0).sum())}/{len(sub)}")
        aggregates.append({"comparison": "standardized − unstandardized", "held fixed": PROTOCOL_NAME[protocol],
                          "model set": set_label, "models": len(sub),
                          "mean Δ F1": delta_f1.mean(), "Δ F1 sd across models": delta_f1.std(),
                          "mean Δ ROC-AUC": delta_auc.mean(),
                          "Δ ROC-AUC sd across models": delta_auc.std(),
                          "models with Δ F1 < 0": int((delta_f1 < 0).sum()), "Spearman": rho})
    display(U.format_sd(table))

# The four comparisons over both model sets, in the order the paper cites them.
aggregates = pd.DataFrame(aggregates)
aggregates.to_csv(out("across_model_aggregates_by_set.csv"), index=False)
print("\\nAcross-model aggregations over both sets (the paper quotes the 22-model rows)")
display(aggregates.round(4))
""")
code("""
fig, axes = plt.subplots(1, len(PROTOCOLS), figsize=(8 * len(PROTOCOLS), 9), sharex=True)
for ax, protocol in zip(axes, PROTOCOLS):
    test = all_means.query("protocol == @protocol and part == 'test'")
    order = dataset_comparison[protocol].index[::-1]
    positions = np.arange(len(order))
    for offset, dataset, color in ((-0.2, "standardized", "tab:green"), (0.2, "unstandardized", "tab:gray")):
        values_ = test.query("dataset == @dataset").set_index("model").loc[order]
        ax.barh(positions + offset, values_["f1_mean"], height=0.4, xerr=values_["f1_std"],
                color=color, label=DATASET_NAME[dataset], error_kw={"lw": 0.8})
    ax.set_yticks(positions)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("test F1 (mean ± sd over the 15 folds)")
    ax.set_title(f"Standardized vs unstandardized — {PROTOCOL_NAME[protocol]}")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(out("standardized_vs_unstandardized.png"), dpi=150, bbox_inches="tight")
plt.show()
""")
code("""
# Summary: best individual and best ensemble of each setting (test F1, mean ± sd over 15 folds)
rows = []
for dataset, protocol in SETTINGS:
    test = all_means.query("dataset == @dataset and protocol == @protocol and part == 'test'").set_index("model")
    individual = test.drop(U.ALL_ENSEMBLES, errors="ignore")["f1_mean"].idxmax()
    ensemble = test.loc[[m for m in U.ALL_ENSEMBLES if m in test.index], "f1_mean"].idxmax()
    rows.append({"dataset": DATASET_NAME[dataset], "protocol": PROTOCOL_NAME[protocol],
                   "best individual": individual,
                   "F1 of the best individual": test.loc[individual, "f1_mean"],
                   "F1 of the best individual sd": test.loc[individual, "f1_std"],
                   "best ensemble": ensemble,
                   "F1 of the best ensemble": test.loc[ensemble, "f1_mean"],
                   "F1 of the best ensemble sd": test.loc[ensemble, "f1_std"]})
scenario_summary = pd.DataFrame(rows)
scenario_summary.to_csv(out("ex033_resumo_cenarios.csv"), index=False)
display(U.format_sd(scenario_summary))
""")

# ---------------------------------------------------------------------------- 12
md("""
## 12. External validation

The external set (1,557 compounds, 35 positives) takes part in no fit and no decision anywhere
in this notebook. It is scored a single time, with no k-fold, by models refitted on the entire
training set:

- **Models:** the best individual and the best ensemble under the scaffold split (highest mean test F1), in each dataset. The final models follow section 3.1: `C`, greedy members, weights and meta-model come from a 5-fold scaffold CV over the full training set. The unstandardized dataset gets its own final models here, fitted the same way.
- **Two versions of the external set:**
  - *unstandardized*: the unstandardized SMILES, scored by the unstandardized models;
  - *standardized*: the SMILES go through the same standardization as section 1 (forms of one substance become a single row, positive if any form is positive), with embeddings generated for them, and are scored by the standardized models.
- **Overlap removed:** external compounds that are the same substance (same standardized InChIKey) as a training compound leave the evaluation, because the models have seen them.
- **Cuts:** *full* (everything but the overlap) and *novel scaffold* (only compounds whose Bemis–Murcko skeleton is absent from the training set).
- **Screening metrics**, all threshold-free: Precision@K and Recall@K for K = 50, 100 and 200, and BEDROC (α = 20). The external prevalence (≈2%) is far below the training one (17%), so these metrics compare the ordering of the compounds rather than a cut at 0.5.

The predictions are cached as a CSV. Re-running loads that file and does not read the external set again; every new read is appended to the external-read log.
""")
code("""
EXTERNAL_CSV = os.path.join(ROOT, "data", "gerollm_reconstructed", "external_test_set.csv")
EXTERNAL_STANDARDIZED_CSV = os.path.join(STANDARDIZED_DIR, "external_standardized.csv")
EXTERNAL_EMBEDDINGS_DIR = {
    "unstandardized": os.path.join(ROOT, "data", "embeddings", "unstandardized_external"),
    "standardized": os.path.join(ROOT, "data", "embeddings", "standardized_external"),
}
FINAL_MODELS_UNSTANDARDIZED = os.path.join(RESULTS_DIR, "models", "final_models_unstandardized.joblib")
EXTERNAL_PREDICTIONS_CSV = out("external_predictions.csv")   # cached: the final models have not changed
EXTERNAL_READ_LOG = os.path.join(RESULTS_DIR, "logs", "external_read_log.json")

# The two versions of the external set and the chemistry of each (InChIKey and scaffold group)
external = {"unstandardized": pd.read_csv(EXTERNAL_CSV)}
if os.path.exists(EXTERNAL_STANDARDIZED_CSV):
    external["standardized"] = pd.read_csv(EXTERNAL_STANDARDIZED_CSV)
else:
    external["standardized"], _, _ = U.standardize_dataset(external["unstandardized"])
    external["standardized"].to_csv(EXTERNAL_STANDARDIZED_CSV, index=False)
training_chemistry = {"unstandardized": chemistry_unstandardized, "standardized": chemistry}
external_chemistry = {b: U.chemistry_table(external[b]["smiles"].astype(str).tolist()) for b in DATASETS}

cuts = {b: U.external_cuts(training_chemistry[b], external_chemistry[b]) for b in DATASETS}
display(pd.DataFrame({DATASET_NAME[b]: {
    "compounds in the external set": len(external[b]),
    "positives in the external set": int(external[b]["label"].sum()),
    "same substance as training (removed)": cuts[b]["n_same_substance_removed"],
    "full cut": int(cuts[b]["full"].sum()),
    "positives in the full cut": int(external[b]["label"].to_numpy()[cuts[b]["full"]].sum()),
    "novel-scaffold cut": int(cuts[b]["novel_scaffold"].sum()),
    "positives in the novel-scaffold cut": int(external[b]["label"].to_numpy()[cuts[b]["novel_scaffold"]].sum()),
} for b in DATASETS}))
""")
code("""
# Final models of the unstandardized dataset (the standardized ones come from section 3.1)
if os.path.exists(FINAL_MODELS_UNSTANDARDIZED):
    final_unstandardized = joblib.load(FINAL_MODELS_UNSTANDARDIZED)
    print(f"[checkpoint] final models loaded from {os.path.basename(FINAL_MODELS_UNSTANDARDIZED)}")
else:
    final_unstandardized = U.fit_final_models(X["unstandardized"], Y["unstandardized"], NAMES, BINARY_FPS, C_GRID,
                                       GROUPS["unstandardized"], n_jobs=N_JOBS)
    joblib.dump(final_unstandardized, FINAL_MODELS_UNSTANDARDIZED)
final_models_by_dataset = {"standardized": final_models, "unstandardized": final_unstandardized}

# Best individual and best ensemble of the scaffold split, in each dataset
evaluated = {}
for dataset in DATASETS:
    test = all_means.query("dataset == @dataset and protocol == 'scaffold' and part == 'test'").set_index("model")["f1_mean"]
    evaluated[dataset] = [best_individual[dataset], test[[m for m in U.ENSEMBLES if m in test.index]].idxmax()]
    print(f"{DATASET_NAME[dataset]}: {evaluated[dataset]}; members of the final greedy ensemble: "
          f"{final_models_by_dataset[dataset]['membros_gulosos']}")
""")
code("""
# External embeddings: only the representations the evaluated models use
needed = {b: sorted({n for m in evaluated[b] for n in U.encoders_of_model(final_models_by_dataset[b], m)}) for b in DATASETS}


def external_file(dataset, chave):
    if dataset == "unstandardized" and chave == "ecfp4":      # this cut stores ECFP4 under another name
        return os.path.join(EXTERNAL_EMBEDDINGS_DIR[dataset], "fingerprints.npy")
    return os.path.join(EXTERNAL_EMBEDDINGS_DIR[dataset], f"Embeddings_{chave}.npy")


def external_ready(dataset, name):
    file_path = external_file(dataset, ENCODERS[name])
    return os.path.exists(file_path) and np.load(file_path, mmap_mode="r").shape[0] == len(external[dataset])


missing = [n for n in needed["standardized"] if not external_ready("standardized", n)]
if missing:
    command = [EMBEDDINGS_PYTHON, EMBEDDINGS_SCRIPT, "--csv", EXTERNAL_STANDARDIZED_CSV, "--coluna", "smiles",
               "--set_label", "label", "--saida", EXTERNAL_EMBEDDINGS_DIR["standardized"], "--encoders", *missing]
    print("Generating embeddings for the standardized external set:", ", ".join(missing))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for row in process.stdout:
        if row.startswith(("[ok]", "[erro]", "[checkpoint]", "Resumo")):
            print(row, end="")
    process.wait()

X_external = {}
for dataset in DATASETS:
    absent = [n for n in needed[dataset] if not external_ready(dataset, n)]
    assert not absent, f"external embeddings missing ({dataset}): {absent}"
    X_external[dataset] = {n: np.load(external_file(dataset, ENCODERS[n])).astype(np.float32) for n in needed[dataset]}
    for n in needed[dataset]:        # same order and same labels as the CSV
        folder, file_name = os.path.split(external_file(dataset, ENCODERS[n]))
        labels_name = ("labels_fingerprints.npy" if file_name == "fingerprints.npy"
                        else file_name.replace("Embeddings_", "labels_"))
        labels = np.load(os.path.join(folder, labels_name)).astype(int)
        assert (labels == external[dataset]["label"].to_numpy()).all(), (dataset, n)
    print(f"{DATASET_NAME[dataset]}: {len(needed[dataset])} external representations loaded")
""")
code("""
# Single read of the external set: the predictions are cached and never recomputed
if os.path.exists(EXTERNAL_PREDICTIONS_CSV):
    external_predictions = pd.read_csv(EXTERNAL_PREDICTIONS_CSV)
    print(f"[checkpoint] external predictions loaded from {os.path.basename(EXTERNAL_PREDICTIONS_CSV)} (no new read)")
else:
    parts = []
    for dataset in DATASETS:
        part = pd.DataFrame({"dataset": dataset, "row": np.arange(len(external[dataset])),
                              "label": external[dataset]["label"].astype(int).to_numpy(),
                              "full_cut": cuts[dataset]["full"],
                              "novel_scaffold_cut": cuts[dataset]["novel_scaffold"]})
        for model in evaluated[dataset]:
            part[model] = U.predict_with_model(final_models_by_dataset[dataset], X_external[dataset], model)
        parts.append(part)
    external_predictions = pd.concat(parts, ignore_index=True)
    external_predictions.to_csv(EXTERNAL_PREDICTIONS_CSV, index=False)
    U.log_external_read(EXTERNAL_READ_LOG, "final-models",
                        note="best individual and best ensemble of the scaffold split, final models "
                             "of both datasets; external set in both its standardized and "
                             "unstandardized versions; single read, screening metrics")

rows = []
for dataset in DATASETS:
    part = external_predictions.query("dataset == @dataset")
    for corte, coluna in (("full", "full_cut"), ("novel scaffold", "novel_scaffold_cut")):
        selected = part[part[coluna]]
        for model in evaluated[dataset]:
            rows.append({"external set": DATASET_NAME[dataset].replace("dataset", "external set"), "cut": corte,
                           "model": model, **U.screening_metrics(selected["label"].to_numpy(), selected[model].to_numpy())})
external_screening = pd.DataFrame(rows)
external_screening.to_csv(out("external_screening.csv"), index=False)
print("External set, single read (no k-fold)")
display(external_screening.round(4))
""")

# ---------------------------------------------------------------------------- 13
md("""
## 13. Confidence intervals and paired tests (limitation i)

Up to here every number is a mean ± standard deviation over the 15 folds. That spread says how
much the result moves when the partition moves; it is not an interval for the estimate, because
folds of a repeated CV reuse the same training rows and are therefore dependent. Two
replacements are computed in this section:

- **Scaffold bootstrap.** Within one repetition (seed 42) the 5 folds cover the whole dataset and every compound is predicted exactly once, by a model that has not seen it. The bootstrap resamples **whole scaffold groups** rather than compounds: compounds sharing a skeleton are not independent observations, and resampling row by row treats a congeneric series as several independent draws, which narrows the interval without adding information. 2,000 replicates.
- **Paired bootstrap.** Two overlapping intervals do **not** imply that the difference is null when the models are correlated — and models scored on the same compounds always are. "Does A beat B" is answered by the interval of the difference, with the same resample applied to both score vectors.
- **Corrected resampled t-test (Nadeau–Bengio).** Over the 15 folds, for the claims that exist only as per-fold values. The correction replaces the 1/n variance factor by 1/n + n_test/n_train = 0.3167. The uncorrected p-value is printed alongside, to show the size of the difference.

The setting is the strictest one: **standardized dataset, scaffold split**. The reference is the best individual representation.
""")
code("""
N_BOOT = 2000
CI_MODELS = U.ALL_ENSEMBLES + [best_individual["standardized"]]
ci_setting = ("standardized", "scaffold")

proba_oof = U.oof_of_one_repetition(
    Y["standardized"], indices[ci_setting],
    {**members[ci_setting], **ensembles[ci_setting]}, CI_MODELS, seed=42)

intervals = U.intervals_table(Y["standardized"], proba_oof, GROUPS["standardized"],
                                 CI_MODELS, n_boot=N_BOOT)
intervals.to_csv(out("bootstrap_intervals_scaffold_standardized.csv"), index=False)
wide = intervals.assign(
    value=lambda d: d.apply(lambda r: f"{r['point']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]", axis=1)
).pivot(index="model", columns="metric", values="value")
print(f"95% CI by scaffold bootstrap ({N_BOOT} replicates), standardized dataset, scaffold split, seed 42")
display(wide)
""")
code("""
# Each ensemble against the best individual, with the same resample applied to both
paired = U.paired_table(Y["standardized"], proba_oof, GROUPS["standardized"],
                           U.ALL_ENSEMBLES, best_individual["standardized"], n_boot=N_BOOT)
paired.to_csv(out("paired_bootstrap_scaffold_standardized.csv"), index=False)
summary = paired.assign(
    value=lambda d: d.apply(lambda r: f"{r['difference']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                                      + ("*" if r["excludes_zero"] else ""), axis=1)
).pivot(index="model", columns="metric", values="value")
print(f"Ensemble − {best_individual['standardized']}, paired scaffold bootstrap. "
      f"* = the interval excludes zero.")
display(summary)
""")
code("""
# Nadeau-Bengio over the 15 folds, in all four settings
per_fold = pd.concat([individual_metrics, ensemble_metrics], ignore_index=True)
rows = []
for dataset, protocol in SETTINGS:
    part = per_fold.query("dataset == @dataset and protocol == @protocol")
    table = U.nadeau_bengio_table(part, U.ALL_ENSEMBLES, best_individual[dataset])
    rows.append(table.assign(dataset=DATASET_NAME[dataset], protocol=PROTOCOL_NAME[protocol]))
nadeau = pd.concat(rows, ignore_index=True)
nadeau.to_csv(out("nadeau_bengio_ttest.csv"), index=False)
print("Corrected resampled t-test over the 15 folds (uncorrected p in parentheses)")
display(nadeau.assign(
    p=lambda d: d.apply(lambda r: f"{r['corrected p']:.3f} ({r['uncorrected p']:.3f})", axis=1)
).pivot_table(index=["dataset", "protocol", "model"], columns="metric", values="p", aggfunc="first"))
""")

# ---------------------------------------------------------------------------- 14
md("""
## 14. Stacking calibration and threshold-free selection (limitations vi and v)

### 14.1 What calibration changes

Calibration is monotone, so the stacking ROC-AUC and average precision **do not change**: the ordering of the compounds is the same. What changes is where the threshold 0.5 falls in the score distribution, and therefore every metric that depends on the cut.

Here is the part the first diagnosis did not anticipate: **calibrating and still reading the model at 0.5 lowers F1** instead of raising it. Uncalibrated stacking pushes probabilities upward, so at 0.5 it predicts positive too often (high recall, low precision). Calibration shrinks the probabilities toward the real prevalence, 17%; almost nothing then clears 0.5, recall collapses and F1 falls. Both readings are symptoms of the same thing: **0.5 is not the right cut for this model**, before or after.

So this section evaluates the two together. For each stacking variant, F1 is measured at 0.5 and at the **threshold chosen inside the fold** — the one that maximizes F1 on the inner out-of-fold probabilities of the training part, over a wide grid from 0.05 to 0.95. The 0.40–0.60 grid of section 10 is useless here: the optimum of a calibrated model sits far below it. The test fold never enters the choice.

The prediction to check in the table: because calibration is monotone, **the three stackings should reach the same F1 once the threshold is chosen in the fold**. If they do, the diagnosis "uncalibrated stacking" becomes "stacking read at the wrong threshold", and calibration is then worth what it actually delivers — an interpretable probability and a well-placed threshold — rather than extra performance.

### 14.2 Greedy selection: F1 at 0.5 versus average precision

The F1 variant picks its members by maximizing F1 at threshold 0.5 on the very out-of-fold probabilities that later feed the threshold study — which is why that ensemble's out-of-fold curve peaks exactly at 0.5 and the study cannot evaluate any other threshold for it. The AP variant uses no threshold at all in the selection. Comparing the two says whether the circularity was costing performance and whether the selected members change.
""")
code("""
ci_setting = ("standardized", "scaffold")
targets = ["Stacking", "Stacking (isotonic)", "Stacking (Platt)",
         "Greedy (F1)", "Greedy (AP)"]
table = ensemble_means.query("dataset == 'standardized' and protocol == 'scaffold' and part == 'test'").set_index("model")
display(pd.DataFrame({U.METRIC_NAMES[m]: U.mean_sd(table.loc[targets, f"{m}_mean"],
                                                      table.loc[targets, f"{m}_std"])
                      for m in U.METRICS}))

# Fraction of compounds called positive at threshold 0.5: the symptom of miscalibration
rows = []
for name in targets:
    fractions = [( ensembles[ci_setting][(s, k, name)]["test_proba"] >= U.DEFAULT_THRESHOLD).mean()
               for (s, k) in indices[ci_setting]]
    rows.append({"model": name, "fraction predicted positive (test)": np.mean(fractions),
                   "fraction predicted positive sd": np.std(fractions, ddof=1)})
fraction_table = pd.DataFrame(rows)
fraction_table.to_csv(out("calibration_scaffold_standardized.csv"), index=False)
print(f"Real prevalence: {Y['standardized'].mean():.2%}")
display(U.format_sd(fraction_table))
""")
code("""
# Threshold chosen inside the fold, wide grid: this is what fixes the miscalibration
per_fold_threshold = U.in_fold_threshold(Y["standardized"], indices[ci_setting],
                                   ensembles[ci_setting], targets)
per_fold_threshold.to_csv(out("in_fold_threshold_scaffold_standardized.csv"), index=False)

aggregated = per_fold_threshold.groupby("model").agg(["mean", "std"])
columns_ = ["selected threshold"] + [f"{m}: {when}" for m in ("f1", "precision", "recall")
                                    for when in ("0.5", "in-fold threshold")]
table = pd.DataFrame({c: U.mean_sd(aggregated[(c, "mean")], aggregated[(c, "std")])
                       for c in columns_}).loc[targets]
print("Threshold 0.5 against the threshold chosen in the fold (mean ± sd over the 15 folds)")
display(table)
print("F1 gained by choosing the threshold in the fold:")
display((per_fold_threshold.assign(gain=lambda d: d["f1: in-fold threshold"] - d["f1: 0.5"])
         .groupby("model")["gain"].agg(["mean", "std"]).loc[targets].round(4)))
""")
code("""
# Members selected by each criterion, over the 15 folds
for dataset in DATASETS:
    setting = (dataset, "scaffold")
    freq = pd.concat({
        "F1 at 0.5": U.selection_frequency(ensembles[setting], NAMES, "Greedy (F1)")["frequency"],
        "average precision": U.selection_frequency(ensembles[setting], NAMES, "Greedy (AP)")["frequency"],
    }, axis=1)
    freq.to_csv(out(f"greedy_selection_frequency_by_criterion_{dataset}.csv"))
    sizes = {c: [len(r["members"]) for (s, k, m), r in ensembles[setting].items() if m == c]
                for c in ("Greedy (F1)", "Greedy (AP)")}
    print(f"{DATASET_NAME[dataset]}, scaffold — median subset size: "
          + "; ".join(f"{c} {np.median(v):.0f} (from {min(v)} to {max(v)})" for c, v in sizes.items()))
    display(freq.sort_values("average precision", ascending=False).round(3))
""")
code("""
# The threshold study redone for Greedy (AP), which has no peak forced at 0.5
rows = []
fig, axes = plt.subplots(1, len(DATASETS), figsize=(8 * len(DATASETS), 4.5), sharey=True)
for ax, dataset in zip(axes, DATASETS):
    setting = (dataset, "scaffold")
    for model, color in (("Greedy (F1)", "tab:red"), ("Greedy (AP)", "tab:green")):
        probas = {(s, k): r for (s, k, n), r in ensembles[setting].items() if n == model}
        curve, threshold = U.threshold_sweep(Y[dataset], indices[setting], probas)[1:]
        ax.plot(curve.index, curve["f1_oof_mean"], ls="-", color=color, label=f"{model} — out-of-fold")
        ax.plot(curve.index, curve["f1_test_mean"], ls="--", color=color,
                label=f"{model} — test (reference only)")
        ax.scatter([threshold], [curve.loc[threshold, "f1_oof_mean"]], s=160, marker="*", color=color,
                   edgecolors="black", linewidths=0.6, zorder=4)
        rows.append({"dataset": DATASET_NAME[dataset], "criterion": model,
                       "selected threshold (out-of-fold peak)": threshold,
                       "test F1 at 0.5": curve.loc[0.5, "f1_test_mean"],
                       "test F1 at the selected threshold": curve.loc[threshold, "f1_test_mean"]})
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("threshold")
    ax.set_title(f"Threshold by selection criterion — {DATASET_NAME[dataset]}, scaffold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("mean F1 (15 folds)")
fig.tight_layout()
fig.savefig(out("threshold_by_criterion_scaffold.png"), dpi=150, bbox_inches="tight")
plt.show()

threshold_by_criterion = pd.DataFrame(rows)
threshold_by_criterion.to_csv(out("threshold_by_selection_criterion.csv"), index=False)
display(threshold_by_criterion.round(4))
""")

# ---------------------------------------------------------------------------- 15
md("""
## 15. InChIKey collapse (limitation iii)

Under the scaffold split, duplicate leakage is already almost zero (0.4 compounds per fold), and the standardized dataset still loses about 0.016 F1 to the unstandardized one. The most plausible explanation is **counting**: in the unstandardized dataset a substance present in several forms is scored several times in the same test fold, and if those substances are the most studied ones, and therefore the easier ones, F1 rises without the model being any better.

The test is direct and refits nothing: take the test predictions of the **unstandardized** dataset and aggregate them by standardized InChIKey, with one prediction per substance (the mean of the probabilities) and the label by the same "positive if any form is positive" rule used by standardization. If the Δ against the standardized dataset shrinks, the drop was a counting effect. If it holds, the cause is elsewhere.
""")
code("""
unstandardized_keys = chemistry_unstandardized["L5_inchikey"].to_numpy()
COLLAPSE_MODELS = U.ALL_ENSEMBLES + [best_individual["unstandardized"], best_individual["standardized"]]
COLLAPSE_MODELS = list(dict.fromkeys(COLLAPSE_MODELS))

rows = []
for seed in SEEDS:
    proba_unstandardized = U.oof_of_one_repetition(
        Y["unstandardized"], indices[("unstandardized", "scaffold")],
        {**members[("unstandardized", "scaffold")], **ensembles[("unstandardized", "scaffold")]},
        COLLAPSE_MODELS, seed=seed)
    y_collapsed, proba_collapsed = U.collapse_by_inchikey(Y["unstandardized"], proba_unstandardized, unstandardized_keys)
    proba_standardized = U.oof_of_one_repetition(
        Y["standardized"], indices[("standardized", "scaffold")],
        {**members[("standardized", "scaffold")], **ensembles[("standardized", "scaffold")]},
        COLLAPSE_MODELS, seed=seed)
    for model in COLLAPSE_MODELS:
        for setting, y_c, p_c in (("unstandardized (rows)", Y["unstandardized"], proba_unstandardized[model]),
                                  ("unstandardized (collapsed)", y_collapsed, proba_collapsed[model]),
                                  ("standardized", Y["standardized"], proba_standardized[model])):
            rows.append({"seed": seed, "model": model, "evaluation": setting,
                           "compounds": len(y_c), **U.metrics(y_c, p_c)})
collapse = pd.DataFrame(rows)
collapse.to_csv(out("inchikey_collapse_per_seed.csv"), index=False)

collapse_summary = collapse.groupby(["model", "evaluation"])[["compounds"] + U.METRICS].agg(["mean", "std"])
table = pd.DataFrame({c: U.mean_sd(collapse_summary[(c, "mean")], collapse_summary[(c, "std")])
                       for c in ["f1", "roc_auc", "average_precision"]})
table.columns = [U.METRIC_NAMES[c] for c in ["f1", "roc_auc", "average_precision"]]
table.insert(0, "compounds", collapse_summary[("compounds", "mean")].round(0).astype(int))
print("Out-of-fold predictions of the scaffold split, mean ± sd over the 3 repetitions")
display(table)
""")
code("""
# The Δ standardized − unstandardized, before and after the collapse
wide = collapse.pivot_table(index=["model", "seed"], columns="evaluation", values="f1")
deltas = pd.DataFrame({
    "Δ F1 (standardized − rows)": wide["standardized"] - wide["unstandardized (rows)"],
    "Δ F1 (standardized − collapsed)": wide["standardized"] - wide["unstandardized (collapsed)"],
}).groupby("model").agg(["mean", "std"])
table = pd.DataFrame({c: U.mean_sd(deltas[(c, "mean")], deltas[(c, "std")])
                       for c in ["Δ F1 (standardized − rows)", "Δ F1 (standardized − collapsed)"]})
table.to_csv(out("inchikey_collapse_delta.csv"))
print("If the second column is near zero, the standardization drop was a counting effect.")
display(table)
""")

# ---------------------------------------------------------------------------- 16
md("""
## 16. Label-conflict ablation (limitation iv)

Merging by InChIKey joins forms of the same substance that arrived with different labels. There are **82 such substances**, all resolved as positive by the same rule the dataset already used for conflicts between sources. That choice had never been tested. Here the three policies are compared:

| Policy | Compounds | Positives | Prevalence |
|---|---|---|---|
| positive (the default) | 3,752 | 637 | 16.98% |
| removed | 3,670 | 555 | 15.12% |
| negative | 3,752 | 555 | 14.79% |

Unlike every other section, this one **refits the models**: changing labels invalidates the checkpoints. To keep the cost near an hour, the ablation runs on the standardized dataset and the scaffold split only, which is the reference setting; that is 2 new policies × 18 representations × 15 folds = 540 fits, in checkpoints of their own. There is no ablation for the unstandardized dataset, where these forms are separate rows with their own labels and there is no conflict to resolve.

What matters is not the absolute value, which moves with prevalence alone, but whether **the ordering of the representations and the ensemble advantage survive** the label policy.
""")
code("""
ablation_checkpoints = U.Checkpoints(os.path.join(RESULTS_DIR, "checkpoints", "label_ablation"))
ablation = {}
for policy in U.CONFLICT_POLICIES:
    policy_dataset, kept = U.apply_conflict_policy(standardized, policy)
    y_policy = policy_dataset["label"].astype(int).to_numpy()
    groups_policy = GROUPS["standardized"][kept]
    X_policy = {n: X["standardized"][n][kept] for n in NAMES}
    folds_policy = U.repeated_scaffold_folds(y_policy, groups_policy, n_splits=N_SPLITS, seeds=SEEDS)
    indices_policy = {(s, k): (tr, te) for s, k, tr, te in folds_policy}
    print(f"[{policy}] {len(y_policy)} compounds, {y_policy.sum()} positives ({y_policy.mean():.2%})")
    # The "positive" policy is the default one: same labels, same folds, same checkpoints.
    # Only the other two need new fits (2 x 18 x 15 = 540).
    reuses = policy == "positive"
    members_policy = U.fit_members(
        X_policy, y_policy, folds_policy, NAMES, BINARY_FPS, C_GRID,
        checkpoints if reuses else ablation_checkpoints, "scaffold", groups=groups_policy,
        n_jobs=N_JOBS, prefix="standardized_scaffold" if reuses else f"ablation_{policy}")
    ensembles_policy = {}
    for (seed, k), (tr, te) in indices_policy.items():
        records = {n: members_policy[(seed, k, n)] for n in NAMES}
        for name, record in U.fold_ensembles(y_policy, tr, records, NAMES, seed).items():
            ensembles_policy[(seed, k, name)] = record
    ablation[policy] = {"y": y_policy, "indices": indices_policy,
                         "metrics": pd.concat([
                             U.metrics_table(y_policy, indices_policy, members_policy, "scaffold", policy),
                             U.metrics_table(y_policy, indices_policy, ensembles_policy, "scaffold", policy)],
                             ignore_index=True)}

ablation_metrics = pd.concat([ablation[p]["metrics"] for p in U.CONFLICT_POLICIES], ignore_index=True)
ablation_metrics.to_csv(out("label_ablation_per_fold.csv"), index=False)
ablation_means = U.mean_by_model(ablation_metrics)
ablation_means.to_csv(out("label_ablation_mean.csv"), index=False)
""")
code("""
# Does the ordering of the representations change with the policy?
ablation_test = ablation_means.query("part == 'test'")
orders = {p: ablation_test.query("dataset == @p").set_index("model")["f1_mean"] for p in U.CONFLICT_POLICIES}
rows = []
for metric_ in ("f1", "roc_auc", "average_precision"):
    values_ = {p: ablation_test.query("dataset == @p").set_index("model")[f"{metric_}_mean"]
               for p in U.CONFLICT_POLICIES}
    common = sorted(set.intersection(*(set(v.index) for v in values_.values())))
    for a, b in (("positive", "removed"), ("positive", "negative")):
        rho = spearmanr(values_[a].loc[common], values_[b].loc[common]).statistic
        rows.append({"metric": U.METRIC_NAMES[metric_], "comparison": f"{a} vs {b}",
                       "Spearman": rho})
ablation_spearman = pd.DataFrame(rows).pivot(index="metric", columns="comparison", values="Spearman")
ablation_spearman.to_csv(out("label_ablation_spearman.csv"))
print(f"Spearman correlation between the rankings of the {len(common)} models under each policy")
display(ablation_spearman.round(3))
""")
code("""
# Does the ensemble advantage survive every policy?
rows = []
for policy in U.CONFLICT_POLICIES:
    t = ablation_test.query("dataset == @policy").set_index("model")
    individuals = t.drop(U.ALL_ENSEMBLES, errors="ignore")
    best_ind = individuals["f1_mean"].idxmax()
    for metric_ in ("f1", "roc_auc", "average_precision", "mcc"):
        best_ens = t.loc[[m for m in U.ALL_ENSEMBLES if m in t.index], f"{metric_}_mean"].idxmax()
        rows.append({"policy": policy, "metric": U.METRIC_NAMES[metric_],
                       "best individual": best_ind,
                       "best ensemble": best_ens,
                       "Δ (ensemble − individual)": t.loc[best_ens, f"{metric_}_mean"]
                                                    - individuals[f"{metric_}_mean"].max()})
ablation_gain = pd.DataFrame(rows)
ablation_gain.to_csv(out("label_ablation_ensemble_gain.csv"), index=False)
print("If Δ is positive under all three policies, the ensemble advantage does not depend on the label rule.")
display(ablation_gain.pivot(index="policy", columns="metric", values="Δ (ensemble − individual)").round(4))
display(ablation_gain)
""")

# =========================================================================================
for position, cell in enumerate(cells):
    cell["id"] = f"benchmark-{position:03d}"
    cell["source"] = cell["source"].splitlines(keepends=True)

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python", "version": "3.14"}},
    "nbformat": 4, "nbformat_minor": 5,
}
destination = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "notebooks", NOTEBOOK))
with open(destination, "w") as file_path:
    json.dump(notebook, file_path, indent=1, ensure_ascii=False)
print(f"written {destination} ({len(cells)} cells)")
