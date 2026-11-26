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

NOTEBOOK = "EX_032_033_benchmark_completo.ipynb"
cells = []


def md(texto):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n")})


def code(texto):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": texto.strip("\n")})


# =========================================================================================
md("""
# Combining Frozen Molecular Representations for Geroprotector Prediction Across Four Evaluation Protocols

This notebook is the whole benchmark behind the paper: 18 pretrained molecular representations used as frozen feature extractors, each with an L2-regularized logistic regression, plus five ensembles, evaluated in four paired settings — the chemically standardized dataset and the conventional canonical-SMILES dataset, each under a Bemis–Murcko scaffold split and a random stratified split, 5 folds × 3 seeds = 15 outer folds each.

Sections 1 to 12 are the benchmark. Sections 13 to 16 are the analyses that close five of the eight limitations the paper lists: bootstrap confidence intervals and paired tests, stacking calibration and threshold-free member selection, the InChIKey collapse, and the label-conflict ablation. Nothing is refitted between the two parts: the same 18 encoders, the same 15 folds and the same 1,080 member checkpoints are reused throughout.

**Two model sets, and why both are reported.** Sections 7, 8 and 11 aggregate *across models*: the mean Δ per model and the Spearman correlation between rankings. The result depends on which models enter the average, and this study grew by three ensembles along the way — Greedy (AP), Platt-calibrated stacking and isotonic-calibrated stacking. The paper quotes the **22-model** set (18 representations + 4 ensembles): Δ F1 of +0.042 and +0.0215, Δ ROC-AUC of +0.041 and +0.024, the −0.016 drop with 19 of 22 models declining, and the minimum Spearman of 0.948. Over all 25 models the same quantities are +0.047, +0.024, +0.042, +0.025, −0.021 with 22 of 25 declining, and a minimum Spearman of 0.862. Both are computed, printed side by side and saved to `across_model_aggregates_by_set.csv`. Everything else — per-model metrics, ensembles, confidence intervals, the external set — is model by model and does not depend on the choice.

| Limitation in the paper | Where it is addressed |
|---|---|
| (i) No CI nor paired test on the standardized dataset | Section 13 |
| (iii) Cause of the standardization drop not isolated | Section 15 |
| (iv) The 82 label conflicts never ablated | Section 16 |
| (v) Circularity between greedy selection and the threshold study | Sections 5 and 14 |
| (vi) Uncalibrated stacking | Sections 5 and 14 |

Limitations (ii) MAT and R-MAT, (vii) the external set is no longer unread, and (viii) frozen extractor with a linear head are not addressed here: the first depends on a package unavailable in this environment, and the other two cannot be settled by computation.

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
sys.path.insert(0, os.path.join(ROOT, "utils", "regressao_logistica"))
import ex033_utils as U
U = importlib.reload(U)   # a reused kernel will not pick up module edits without this

BASE_ORIGINAL = os.path.join(ROOT, "dados", "Bases de dados", "Treinamento - limpas",
                             "Base_GeroLLM_reconstruida", "primary_dataset_dedup.csv")
DIR_BASE = os.path.join(ROOT, "dados", "Bases de dados", "Treinamento - limpas",
                        "Base_GeroLLM_padronizada")
BASE_PADRONIZADA = os.path.join(DIR_BASE, "primary_padronizada.csv")
QUIMICA_ORIGINAL = os.path.join(DIR_BASE, "padronizacao_linhas_originais.csv")
QUIMICA_PADRONIZADA = os.path.join(DIR_BASE, "padronizacao_base_padronizada.csv")

# Embedding folders: one .npy per encoder, per dataset.
DIR_EMBEDDINGS = {
    "padronizada": os.path.join(ROOT, "dados", "Embeddings_salvos", "Embeddings_GeroLLM_padronizada"),
    "original": os.path.join(ROOT, "dados", "Embeddings_salvos", "Embeddings_GeroLLM"),
}
SCRIPT_EMBEDDINGS = os.path.join(ROOT, "utils", "embeddings", "gerar_embeddings_base.py")
# The encoders run in the `embeddings` conda environment (torch, transformers,
# torch_geometric, gensim, selfies, unimol_tools); this notebook runs in the analysis one.
PYTHON_EMBEDDINGS = os.path.expanduser("~/miniconda3/envs/embeddings/bin/python")

RESULTS_DIR = os.path.join(ROOT, "resultados")
checkpoints = U.Checkpoints(os.path.join(RESULTS_DIR, "checkpoints", "checkpoints_ex032"))
MODELOS_FINAIS = os.path.join(RESULTS_DIR, "modelos", "ex032_modelos_finais.joblib")


def out(nome):
    pasta = os.path.join(RESULTS_DIR, {".csv": "csv", ".png": "figuras"}[os.path.splitext(nome)[1]])
    os.makedirs(pasta, exist_ok=True)
    return os.path.join(pasta, nome)


# ---- protocol ----------------------------------------------------------------------------
SEEDS = (42, 43, 44)
N_SPLITS = 5
C_GRID = np.logspace(-4, 2, 7)          # 1e-4 .. 1e2, seven values
BASES = ["padronizada", "original"]
NOME_BASE = {"padronizada": "base padronizada", "original": "base não padronizada"}
PROTOCOLOS = ["scaffold", "aleatorio"]
NOME_PROTOCOLO = {"scaffold": "split por scaffold", "aleatorio": "split aleatório estratificado"}
CENARIOS = [(base, protocolo) for base in BASES for protocolo in PROTOCOLOS]
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
BINARIOS = {"ECFP4-2048", "Atom-Pair", "Graph-FP", "TT"}


def arquivos_embedding(base, chave):
    \"\"\"Paths of the embedding matrix and of the labels of one encoder in one dataset.\"\"\"
    pasta = DIR_EMBEDDINGS[base]
    if base == "original" and chave == "ecfp4":      # this cut stores ECFP4 under another name
        return os.path.join(pasta, "fingerprints.npy"), os.path.join(pasta, "rotulos_fingerprints.npy")
    return os.path.join(pasta, f"Embeddings_{chave}.npy"), os.path.join(pasta, f"rotulos_{chave}.npy")
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
os.makedirs(DIR_BASE, exist_ok=True)
original = pd.read_csv(BASE_ORIGINAL)
if os.path.exists(BASE_PADRONIZADA):
    padronizada = pd.read_csv(BASE_PADRONIZADA)
    quimica_original = pd.read_csv(QUIMICA_ORIGINAL)
    print(f"[checkpoint] standardized dataset loaded from {os.path.basename(BASE_PADRONIZADA)}")
else:
    padronizada, quimica_original, _ = U.padronizar_base(original)
    padronizada.to_csv(BASE_PADRONIZADA, index=False)
    quimica_original.to_csv(QUIMICA_ORIGINAL, index=False)

summary = {
    "rows in the unstandardized dataset": len(original),
    "distinct substances (standardized dataset)": len(padronizada),
    "rows merged as duplicates": len(original) - len(padronizada),
    "substances that merged more than one row": int((padronizada["n_formas"] > 1).sum()),
    "substances with conflicting labels (resolved positive)": int(padronizada["rotulo_conflitante"].sum()),
    "positives before / after": f"{int(original['label'].sum())} / {int(padronizada['label'].sum())}",
    "prevalence after": f"{padronizada['label'].mean():.2%}",
}
display(pd.Series(summary, name="value").to_frame())
""")
code("""
# How many duplicates each standardization step revealed
display(U.duplicate_audit(quimica_original))
exemplos = padronizada.query("n_formas > 1").head(8)[["smiles", "smiles_original", "label", "n_formas",
                                                        "rotulo_conflitante", "name"]]
display(exemplos)
""")
code("""
# Scaffold groups, used by the scaffold split. In the unstandardized dataset they are
# computed row by row, before any merging.
if os.path.exists(QUIMICA_PADRONIZADA):
    quimica = pd.read_csv(QUIMICA_PADRONIZADA)
else:
    quimica = U.chemistry_table(padronizada["smiles"].tolist())
    quimica.to_csv(QUIMICA_PADRONIZADA, index=False)
assert len(quimica) == len(padronizada) and len(quimica_original) == len(original)
repetidos = int(quimica["L5_inchikey"].duplicated().sum())
if repetidos:
    print(f"Warning: {repetidos} compounds still collide when the standardized SMILES is re-standardized.")

Y = {"padronizada": padronizada["label"].astype(int).to_numpy(),
     "original": original["label"].astype(int).to_numpy()}
GRUPOS = {"padronizada": quimica["group"].to_numpy(),
          "original": quimica_original["group"].to_numpy()}
for base in BASES:
    print(f"{NOME_BASE[base]}: {len(Y[base])} compounds, {Y[base].sum()} positives "
          f"({Y[base].mean():.2%}), {len(np.unique(GRUPOS[base]))} scaffold groups")
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


def embedding_pronto(base, chave):
    arquivo = arquivos_embedding(base, chave)[0]
    return os.path.exists(arquivo) and np.load(arquivo, mmap_mode="r").shape[0] == len(Y[base])


prontos = [n for n, chave in ENCODERS.items() if embedding_pronto("padronizada", chave)]
faltando = [n for n in ENCODERS if n not in prontos]
print(f"Folder: {DIR_EMBEDDINGS['padronizada']}")
print(f"{len(prontos)} of {len(ENCODERS)} encoders have complete embeddings for the standardized dataset.")

if REGENERATE_EMBEDDINGS is None and prontos:
    resposta = input(f"Regenerate the embeddings? {len(prontos)} ready, "
                     f"{len(faltando)} missing. [y = regenerate all / Enter = reuse] ")
    REGENERATE_EMBEDDINGS = resposta.strip().lower() in ("y", "yes")

a_gerar = list(ENCODERS) if REGENERATE_EMBEDDINGS else faltando
if a_gerar:
    comando = [PYTHON_EMBEDDINGS, SCRIPT_EMBEDDINGS, "--csv", BASE_PADRONIZADA,
               "--coluna", "smiles", "--rotulo", "label", "--saida", DIR_EMBEDDINGS["padronizada"],
               "--encoders", *a_gerar] + (["--force"] if REGENERATE_EMBEDDINGS else [])
    print("Generating:", ", ".join(a_gerar))
    processo = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for linha in processo.stdout:
        if linha.startswith(("[ok]", "[error]", "[checkpoint]", "Summary")):
            print(linha, end="")
    processo.wait()
else:
    print("Reusing the embeddings already on disk.")
""")
code("""
# Only encoders with embeddings in both datasets take part, so that the comparison uses the
# same representations throughout.
NOMES = [n for n, chave in ENCODERS.items() if all(embedding_pronto(b, chave) for b in BASES)]
ausentes = [n for n in ENCODERS if n not in NOMES]
if ausentes:
    print(f"WARNING: {len(ausentes)} encoder(s) without embeddings, dropped from what follows: {ausentes}")

X = {base: {} for base in BASES}
for base in BASES:
    for n in NOMES:
        arquivo, arquivo_rotulos = arquivos_embedding(base, ENCODERS[n])
        X[base][n] = np.load(arquivo).astype(np.float32)
        rotulos = np.load(arquivo_rotulos).astype(int)
        assert X[base][n].shape[0] == len(Y[base]) and (rotulos == Y[base]).all(), (base, n)
display(pd.DataFrame({NOME_BASE[b]: {n: X[b][n].shape[1] for n in NOMES} for b in BASES}).T)
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
for base in BASES:
    folds[(base, "aleatorio")] = U.repeated_random_folds(Y[base], n_splits=N_SPLITS, seeds=SEEDS)
    folds[(base, "scaffold")] = U.repeated_scaffold_folds(Y[base], GRUPOS[base], n_splits=N_SPLITS,
                                                          seeds=SEEDS)
indices = {c: {(s, k): (tr, te) for s, k, tr, te in folds[c]} for c in CENARIOS}

# Scaffold leakage audit: under the scaffold split no skeleton may be in train and test at once
verificacao = pd.concat([U.scaffolds_compartilhados(folds[(b, p)], GRUPOS[b]).assign(
    base=NOME_BASE[b], protocolo=NOME_PROTOCOLO[p]) for b, p in CENARIOS], ignore_index=True)
verificacao.to_csv(out("ex033_verificacao_scaffold.csv"), index=False)
resumo = verificacao.groupby(["dataset", "protocol"])[["scaffolds in test", "scaffolds in train and test"]].agg(["mean", "std", "max"])
print("Scaffolds shared between train and test, per fold (15 folds)")
display(resumo.round(2))
no_scaffold = verificacao[verificacao["protocol"] == NOME_PROTOCOLO["scaffold"]]
assert no_scaffold["scaffolds in train and test"].eq(0).all(), "scaffold leakage"
print("OK: under the scaffold split, no skeleton appears in train and test at the same time.")

membros = {}
for base, protocolo in CENARIOS:
    membros[(base, protocolo)] = U.rodar_membros(
        X[base], Y[base], folds[(base, protocolo)], NOMES, BINARIOS, C_GRID, checkpoints, protocolo,
        grupos=GRUPOS[base] if protocolo == "scaffold" else None, n_jobs=N_JOBS,
        prefixo=f"{base}_{protocolo}")
    print(f"{NOME_BASE[base]}, {protocolo}: {len(membros[(base, protocolo)])} fits "
          f"(15 folds x {len(NOMES)} representations)")

metricas_individuais = pd.concat(
    [U.tabela_metricas(Y[b], indices[(b, p)], membros[(b, p)], p, b) for b, p in CENARIOS],
    ignore_index=True)
metricas_individuais.to_csv(out("ex033_metricas_individuais_por_fold.csv"), index=False)
""")
code("""
def mostrar_por_fold(tabela, modelo):
    \"\"\"Per-fold metrics of one model, train and test, plus the mean and sd over the 15 folds.\"\"\"
    for base, protocolo in CENARIOS:
        parte = tabela.query("base == @base and protocolo == @protocolo and modelo == @modelo")
        largo = parte.pivot_table(index=["seed", "fold"], columns="parte", values=U.METRICAS)
        largo.columns = [f"{U.NOMES_METRICAS[m]} ({p})" for m, p in largo.columns]
        largo = largo[[f"{U.NOMES_METRICAS[m]} ({p})" for m in U.METRICAS for p in ("train", "test")]]
        media, desvio = largo.mean(), largo.std()
        largo.loc[("mean", ""), :] = media
        largo.loc[("standard deviation", ""), :] = desvio
        print(f"{modelo} — {NOME_BASE[base]} — {NOME_PROTOCOLO[protocolo]}")
        display(largo.round(4))


for nome in NOMES:
    mostrar_por_fold(metricas_individuais, nome)
""")
code("""
media_individuais = U.media_por_modelo(metricas_individuais)
media_individuais.to_csv(out("ex033_metricas_individuais_media.csv"), index=False)
for base, protocolo in CENARIOS:
    for parte in ("teste", "treino"):
        selecao = (media_individuais.query("base == @base and protocolo == @protocolo and parte == @parte")
                   .set_index("modelo").sort_values("f1_mean", ascending=False))
        tabela = pd.DataFrame({U.NOMES_METRICAS[m]: U.media_dp(selecao[f"{m}_mean"], selecao[f"{m}_std"])
                               for m in U.METRICAS})
        print(f"Mean ± sd over the 15 folds — {NOME_BASE[base]} — {NOME_PROTOCOLO[protocolo]} — {parte}")
        display(tabela)
""")
md("""
### 3.1 Final models on the full dataset

Each representation is refit on the whole standardized dataset; the unstandardized dataset only serves as a comparison and gets its final models in section 12. The `C` values and the out-of-fold probabilities that feed the final ensembles (weights, meta-model and greedy subset) come from a 5-fold scaffold CV over the full dataset. The models are saved for use in the ensembles and in later predictions.
""")
code("""
if os.path.exists(MODELOS_FINAIS):
    finais = joblib.load(MODELOS_FINAIS)
    print(f"[checkpoint] final models loaded from {os.path.basename(MODELOS_FINAIS)}")
else:
    finais = U.treinar_finais(X["padronizada"], Y["padronizada"], NOMES, BINARIOS, C_GRID,
                              GRUPOS["padronizada"], n_jobs=N_JOBS)
    os.makedirs(os.path.dirname(MODELOS_FINAIS), exist_ok=True)
    joblib.dump(finais, MODELOS_FINAIS)
display(pd.DataFrame({"C": finais["C"], "weight in the weighted vote": finais["pesos_roc_auc"]}).round(4))
print("Members of the final greedy ensemble:", finais["membros_gulosos"])
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
for base, protocolo in CENARIOS:
    cenario = (base, protocolo)
    ensembles[cenario] = {}
    for (seed, k), (tr, te) in indices[cenario].items():
        registros = {n: membros[cenario][(seed, k, n)] for n in NOMES}
        for nome, registro in U.ensembles_do_fold(Y[base], tr, registros, NOMES, seed).items():
            ensembles[cenario][(seed, k, nome)] = registro

metricas_ensembles = pd.concat(
    [U.tabela_metricas(Y[b], indices[(b, p)], ensembles[(b, p)], p, b) for b, p in CENARIOS],
    ignore_index=True)
metricas_ensembles.to_csv(out("ex033_metricas_ensembles_por_fold.csv"), index=False)

for nome in U.ENSEMBLES_033:
    mostrar_por_fold(metricas_ensembles, nome)
""")
code("""
media_ensembles = U.media_por_modelo(metricas_ensembles)
media_ensembles.to_csv(out("ex033_metricas_ensembles_media.csv"), index=False)
for base, protocolo in CENARIOS:
    selecao = media_ensembles.query("base == @base and protocolo == @protocolo").set_index(["parte", "modelo"])
    tabela = pd.DataFrame({f"{U.NOMES_METRICAS[m]} ({p})": U.media_dp(selecao.loc[p, f"{m}_mean"],
                                                                     selecao.loc[p, f"{m}_std"])
                           for m in U.METRICAS for p in ("teste", "treino")})
    print(f"Mean ± sd over the 15 folds — {NOME_BASE[base]} — {NOME_PROTOCOLO[protocolo]}")
    display(tabela)
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
melhor_individual = {}
for base in BASES:
    teste = media_individuais.query("base == @base and protocolo == 'scaffold' and parte == 'teste'")
    melhor_individual[base] = teste.set_index("modelo")["f1_mean"].idxmax()
    print(f"Best individual representation, scaffold split — {NOME_BASE[base]}: {melhor_individual[base]}")

frequencias, consenso = {}, {}
for base in BASES:
    cenario = (base, "scaffold")
    frequencias[base] = U.frequencia_selecao(ensembles[cenario], NOMES)
    tamanhos = [len(r["members"]) for (s, k, m), r in ensembles[cenario].items() if m == "Greedy (F1)"]
    print(f"{NOME_BASE[base]}: {len(tamanhos)} folds, median subset size {np.median(tamanhos):.0f} "
          f"(from {min(tamanhos)} to {max(tamanhos)} members)")
    consenso[base] = frequencias[base].query("frequency >= 0.5").index.tolist()
tabela = pd.concat({NOME_BASE[b]: frequencias[b]["frequency"] for b in BASES}, axis=1)
tabela.to_csv(out("ex033_frequencia_gulosa_scaffold.csv"))
display(tabela.sort_values(NOME_BASE["padronizada"], ascending=False).round(3))
for base in BASES:
    print(f"Consensus members ({NOME_BASE[base]}, selected in >= 50% of the folds): {consenso[base]}")
""")
code("""
linhas = []
for base in BASES:
    for modelo, tipo in ((melhor_individual[base], "best individual"), ("Greedy (F1)", "greedy ensemble")):
        fonte = media_ensembles if modelo == "Greedy (F1)" else media_individuais
        linha = fonte.query("dataset == @base and protocol == 'scaffold' and part == 'test' "
                            "and model == @modelo").iloc[0]
        linhas.append({"dataset": NOME_BASE[base], "kind": tipo, "model": modelo,
                       **{U.NOMES_METRICAS[m]: f"{linha[f'{m}_mean']:.4f} ± {linha[f'{m}_std']:.4f}"
                          for m in U.METRICAS}})
guloso_x_individual = pd.DataFrame(linhas)
guloso_x_individual.to_csv(out("ex033_guloso_x_melhor_individual_scaffold.csv"), index=False)
print("Test folds, scaffold split (mean ± sd over the 15 folds)")
display(guloso_x_individual)
""")
code("""
fig, axes = plt.subplots(1, len(BASES), figsize=(7 * len(BASES), 4), sharey=True)
for ax, base in zip(axes, BASES):
    cenario = (base, "scaffold")
    for (seed, k, nome), registro in ensembles[cenario].items():
        if nome == "Greedy (F1)":
            caminho = registro["path"]
            ax.plot(caminho["n_members"], caminho["f1_inner"], color="tab:blue", alpha=0.35)
    f1_interno_melhor = np.mean([
        f1_score(Y[base][tr], (membros[cenario][(s, k, melhor_individual[base])]["inner_oof"]
                               >= U.LIMIAR_PADRAO).astype(int), zero_division=0)
        for (s, k), (tr, te) in indices[cenario].items()])
    ax.axhline(f1_interno_melhor, color="tab:red", ls="--", lw=1.5,
               label=f"best individual ({melhor_individual[base]})")
    ax.set_title(f"Greedy selection path — {NOME_BASE[base]}, scaffold")
    ax.set_xlabel("number of members")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
axes[0].set_ylabel("inner F1 (training part)")
fig.tight_layout()
fig.savefig(out("ex033_caminho_guloso_scaffold.png"), dpi=150, bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------------------- 7
md("""
## 7. Ranking

Individual models, the three all-representation ensembles and the two greedy ensembles, ordered by mean test F1 over the 15 folds (threshold 0.5), in each dataset and under each split.

The **rank** column is among all 25 models; **rank (paper's 22)** is among the 22 the paper quotes. They differ for anything below one of the three later ensembles: Graph-FP, for instance, is 15th among the 22 and 16th among the 25.
""")
code("""
media_todos = pd.concat([media_individuais, media_ensembles], ignore_index=True)

# Model sets for the across-model aggregations (sections 7, 8 and 11). The paper quotes the
# 22-model numbers; the 25-model set adds the three later ensembles. Both are computed in every
# aggregation, and the set is named in the output.
TODOS_OS_MODELOS = sorted(media_todos["modelo"].unique())
MODELOS_22 = [m for m in TODOS_OS_MODELOS if m not in U.ENSEMBLES_NOVOS]
CONJUNTOS = {"22 (paper)": MODELOS_22, "25 (full)": TODOS_OS_MODELOS}
agregados = []        # filled in sections 8 and 11, saved at the end of section 11
for rotulo, modelos in CONJUNTOS.items():
    print(f"model set {rotulo}: {len(modelos)} models")
assert len(MODELOS_22) == 22 and len(TODOS_OS_MODELOS) == 25, "unexpected model set"

rankings = {}
for base, protocolo in CENARIOS:
    teste = media_todos.query("base == @base and protocolo == @protocolo and parte == 'teste'").set_index("modelo")
    ranking = pd.DataFrame({U.NOMES_METRICAS[m]: U.media_dp(teste[f"{m}_mean"], teste[f"{m}_std"])
                            for m in U.METRICAS})
    ranking["kind"] = ["ensemble" if m in U.ENSEMBLES_033 else "individual" for m in ranking.index]
    ranking = ranking.loc[teste["f1_mean"].sort_values(ascending=False).index]
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    # Rank within the paper's 22 models: that is the one the text cites (Graph-FP, for
    # instance, is 15th among the 22 and 16th among the 25).
    nos_22 = [m for m in ranking.index if m in MODELOS_22]
    ranking.insert(1, "rank (paper's 22)",
                   [nos_22.index(m) + 1 if m in MODELOS_22 else None for m in ranking.index])
    rankings[(base, protocolo)] = ranking
    ranking.to_csv(out(f"ex033_ranking_{base}_{protocolo}.csv"))
    print(f"Ranking by mean test F1 — {NOME_BASE[base]} — {NOME_PROTOCOLO[protocolo]} "
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
comparacao_protocolos = {}
for base in BASES:
    tabela = U.comparar(media_todos.query("base == @base"), "protocolo", "scaffold", "aleatorio")
    tabela = tabela.sort_values("F1 (scaffold)", ascending=False)
    tabela.insert(0, "kind", ["ensemble" if m in U.ENSEMBLES_033 else "individual" for m in tabela.index])
    comparacao_protocolos[base] = tabela
    tabela.to_csv(out(f"scaffold_vs_random_{base}.csv"))
    print(f"{NOME_BASE[base]} — test folds, mean ± sd over the 15 folds")
    for rotulo, modelos in CONJUNTOS.items():
        sub = tabela.loc[[m for m in tabela.index if m in modelos]]
        rho = spearmanr(sub["F1 (scaffold)"], sub["F1 (aleatorio)"]).statistic
        delta_f1 = sub["Δ F1 (aleatorio − scaffold)"]
        delta_auc = sub["Δ ROC-AUC (aleatorio − scaffold)"]
        print(f"  [{len(sub)} models, set {rotulo}] "
              f"mean Δ F1 across models (random − scaffold): {delta_f1.mean():+.4f} ± {delta_f1.std():.4f}; "
              f"mean Δ ROC-AUC: {delta_auc.mean():+.4f} ± {delta_auc.std():.4f}; "
              f"Spearman between the F1 values: {rho:.3f}")
        agregados.append({"comparison": "random − scaffold", "held fixed": NOME_BASE[base],
                          "model set": rotulo, "models": len(sub),
                          "mean Δ F1": delta_f1.mean(), "Δ F1 sd across models": delta_f1.std(),
                          "mean Δ ROC-AUC": delta_auc.mean(),
                          "Δ ROC-AUC sd across models": delta_auc.std(),
                          "models with Δ F1 < 0": int((delta_f1 < 0).sum()), "Spearman": rho})
    display(U.formatar_dp(tabela))
""")
code("""
fig, axes = plt.subplots(1, len(BASES), figsize=(8 * len(BASES), 9), sharex=True)
for ax, base in zip(axes, BASES):
    teste = media_todos.query("base == @base and parte == 'teste'")
    ordem = comparacao_protocolos[base].index[::-1]
    posicoes = np.arange(len(ordem))
    for deslocamento, protocolo, cor in ((-0.2, "scaffold", "tab:blue"), (0.2, "aleatorio", "tab:orange")):
        valores = teste.query("protocolo == @protocolo").set_index("modelo").loc[ordem]
        ax.barh(posicoes + deslocamento, valores["f1_mean"], height=0.4, xerr=valores["f1_std"],
                color=cor, label=NOME_PROTOCOLO[protocolo], error_kw={"lw": 0.8})
    ax.set_yticks(posicoes)
    ax.set_yticklabels(ordem, fontsize=8)
    ax.set_xlabel("test F1 (mean ± sd over the 15 folds)")
    ax.set_title(f"Scaffold vs random — {NOME_BASE[base]}")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(out("ex033_scaffold_x_aleatorio.png"), dpi=150, bbox_inches="tight")
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
for base in BASES:
    cenario = (base, "scaffold")
    membros_div = consenso[base]
    if len(membros_div) < 2:
        print(f"{NOME_BASE[base]}: fewer than 2 consensus members, no diversity map.")
        continue
    acertos = U.acertos_oof(Y[base], indices[cenario], membros[cenario], membros_div, seed=SEEDS[0])
    fig, axes = plt.subplots(1, 4, figsize=(5 * 4, 4.4))
    for ax, (titulo, medida) in zip(axes, U.MEDIDAS_DIVERSIDADE.items()):
        matriz = U.pairwise_matrix(medida, acertos, membros_div)
        imagem = ax.imshow(matriz, cmap="viridis")
        ax.set_xticks(range(len(membros_div)))
        ax.set_xticklabels(membros_div, rotation=90, fontsize=8)
        ax.set_yticks(range(len(membros_div)))
        ax.set_yticklabels(membros_div, fontsize=8)
        for i in range(len(membros_div)):
            for j in range(len(membros_div)):
                if i != j:
                    ax.text(j, i, f"{matriz[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w")
        ax.set_title(titulo)
        fig.colorbar(imagem, ax=ax, fraction=0.046)
    fig.suptitle(f"Diversity of the greedy ensemble — {NOME_BASE[base]}, scaffold split")
    fig.tight_layout()
    fig.savefig(out(f"diversity_{base}_scaffold.png"), dpi=150, bbox_inches="tight")
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
resultados_limiar = []
fig, axes = plt.subplots(1, len(BASES), figsize=(8 * len(BASES), 4.5), sharey=True)
for ax, base in zip(axes, BASES):
    cenario = (base, "scaffold")
    teste = media_todos.query("base == @base and protocolo == 'scaffold' and parte == 'teste'")
    teste = teste.set_index("modelo")["f1_mean"]
    melhor_ensemble = teste[[m for m in U.ENSEMBLES_033 if m in teste.index]].idxmax()

    for modelo, cor in ((melhor_individual[base], "tab:blue"), (melhor_ensemble, "tab:red")):
        fonte = ensembles[cenario] if modelo in U.ENSEMBLES_033 else {
            (s, k, n): {**r, "inner_score": r["inner_oof"]} for (s, k, n), r in membros[cenario].items()}
        probas = {(s, k): r for (s, k, n), r in fonte.items() if n == modelo}
        por_fold, curva, limiar = U.estudo_limiar(Y[base], indices[cenario], probas)
        ax.plot(curva.index, curva["f1_oof_mean"], ls="-", color=cor, label=f"{modelo} — OOF")
        ax.plot(curva.index, curva["f1_test_mean"], ls="--", color=cor,
                label=f"{modelo} — test (reference only)")
        ax.scatter([0.5, 0.5], curva.loc[0.5, ["f1_oof_mean", "f1_test_mean"]], s=70,
                   facecolors="none", edgecolors=cor, linewidths=2, zorder=3)
        ax.scatter([limiar, limiar], curva.loc[limiar, ["f1_oof_mean", "f1_test_mean"]], s=160,
                   marker="*", color=cor, edgecolors="black", linewidths=0.6, zorder=4)
        por_limiar = por_fold.set_index(["threshold", "seed", "fold"]).sort_index()
        linha = {"dataset": NOME_BASE[base], "model": modelo, "selected threshold (out-of-fold peak)": limiar}
        for coluna, nome in (("f1_oof", "out-of-fold F1"), ("f1_test", "test F1")):
            delta = por_limiar.loc[limiar, coluna] - por_limiar.loc[0.5, coluna]
            linha.update({
                f"{nome} at 0.5": curva.loc[0.5, f"{coluna}_mean"],
                f"{nome} at 0.5 sd": curva.loc[0.5, f"{coluna}_std"],
                f"{nome} at the selected threshold": curva.loc[limiar, f"{coluna}_mean"],
                f"{nome} at the selected threshold sd": curva.loc[limiar, f"{coluna}_std"],
                f"Δ {nome}": delta.mean(), f"Δ {nome} sd": delta.std()})
        resultados_limiar.append(linha)
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.scatter([], [], s=70, facecolors="none", edgecolors="gray", linewidths=2,
               label="threshold 0.5")
    ax.scatter([], [], s=160, marker="*", color="gray", edgecolors="black", linewidths=0.6,
               label="selected threshold (out-of-fold peak)")
    ax.set_xlabel("threshold")
    ax.set_title(f"F1 as a function of the threshold — {NOME_BASE[base]}, scaffold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("mean F1 (15 folds)")
fig.tight_layout()
fig.savefig(out("threshold_sweep_scaffold.png"), dpi=150, bbox_inches="tight")
plt.show()

tabela_limiar = pd.DataFrame(resultados_limiar)
tabela_limiar.to_csv(out("threshold_sweep_scaffold.csv"), index=False)
print("Mean ± sd over the 15 folds")
display(U.formatar_dp(tabela_limiar))
""")

# ---------------------------------------------------------------------------- 11
md("""
## 11. Standardized vs unstandardized dataset

First, the eight test metrics of every model in each dataset and under each split (four tables, mean ± standard deviation over the 15 folds, ordered by F1). Then test F1 and ROC-AUC of every model on both datasets, side by side, and the difference standardized − unstandardized. The two datasets do not hold the same compounds (the standardized one merges structural duplicates), so the comparison is between means rather than compound by compound, and the standard deviation of Δ is √(sd_standardized² + sd_unstandardized²). The mean Δ across models comes with the standard deviation across models.

As in section 8, the mean Δ across models and the Spearman are reported for both model sets, and the paper quotes the 22 — including how many models decline under standardization, which is 19 of 22 (and 22 of 25).

In the unstandardized dataset, forms of the same substance (salt and free base, charged and neutral forms, tautomers) are separate rows. Under the random split one form can land in training and another in test, and the model is then scored on a compound it has already seen. The leakage table counts those cases under both splits; under the scaffold split it almost never happens, because forms of the same substance nearly always share a skeleton and fall in the same group.
""")
code("""
for base, protocolo in CENARIOS:
    print(f"Teste — {NOME_BASE[base]} — {NOME_PROTOCOLO[protocolo]} (média ± desvio-padrão das 15 folds)")
    display(rankings[(base, protocolo)])
""")
code("""
chaves = quimica_original["L5_inchikey"].to_numpy()
validas = pd.notna(chaves)
vazamento = []
for protocolo in PROTOCOLOS:
    for seed, k, tr, te in folds[("original", protocolo)]:
        no_treino = set(chaves[tr][validas[tr]])
        vazamento.append({"protocolo": NOME_PROTOCOLO[protocolo], "seed": seed, "fold": k,
                          "compostos de teste com a mesma substância no treino":
                              int(sum(c in no_treino for c in chaves[te][validas[te]])),
                          "compostos de teste": len(te)})
vazamento = pd.DataFrame(vazamento)
vazamento["fração"] = vazamento["compostos de teste com a mesma substância no treino"] / vazamento["compostos de teste"]
colunas = ["compostos de teste com a mesma substância no treino", "compostos de teste", "fração"]
resumo = vazamento.groupby("protocolo")[colunas].agg(["mean", "std"])
print("Vazamento de substâncias duplicadas na base não padronizada (média ± desvio-padrão das 15 folds)")
display(pd.DataFrame({c: U.media_dp(resumo[(c, "mean")], resumo[(c, "std")]) for c in colunas}))
""")
code("""
comparacao_bases = {}
for protocolo in PROTOCOLOS:
    tabela = U.comparar(media_todos.query("protocolo == @protocolo"), "base", "original", "padronizada")
    tabela = tabela.sort_values("F1 (padronizada)", ascending=False)
    tabela.insert(0, "tipo", ["ensemble" if m in U.ENSEMBLES_033 else "individual" for m in tabela.index])
    comparacao_bases[protocolo] = tabela
    tabela.to_csv(out(f"ex033_padronizada_x_original_{protocolo}.csv"))
    print(f"{NOME_PROTOCOLO[protocolo]} — teste, média ± desvio-padrão das 15 folds")
    for rotulo, modelos in CONJUNTOS.items():
        sub = tabela.loc[[m for m in tabela.index if m in modelos]]
        rho = spearmanr(sub["F1 (original)"], sub["F1 (padronizada)"]).statistic
        delta_f1 = sub["Δ F1 (padronizada − original)"]
        delta_auc = sub["Δ ROC-AUC (padronizada − original)"]
        print(f"  [{len(sub)} modelos, conjunto {rotulo}] "
              f"Δ F1 médio entre modelos (padronizada − original): {delta_f1.mean():+.4f} ± {delta_f1.std():.4f}; "
              f"Δ ROC-AUC médio: {delta_auc.mean():+.4f} ± {delta_auc.std():.4f}; "
              f"Spearman entre os F1: {rho:.3f}; "
              f"modelos que caem: {int((delta_f1 < 0).sum())}/{len(sub)}")
        agregados.append({"comparação": "padronizada − original", "fixo": NOME_PROTOCOLO[protocolo],
                          "conjunto": rotulo, "modelos": len(sub),
                          "Δ F1 médio": delta_f1.mean(), "Δ F1 dp entre modelos": delta_f1.std(),
                          "Δ ROC-AUC médio": delta_auc.mean(),
                          "Δ ROC-AUC dp entre modelos": delta_auc.std(),
                          "modelos com Δ F1 < 0": int((delta_f1 < 0).sum()), "Spearman": rho})
    display(U.formatar_dp(tabela))

# As quatro comparações nos dois conjuntos, na ordem em que o artigo as cita.
agregados = pd.DataFrame(agregados)
agregados.to_csv(out("ex033_agregados_por_conjunto.csv"), index=False)
print("\nAgregações entre modelos nos dois conjuntos (o artigo cita as linhas de 22 modelos)")
display(agregados.round(4))
""")
code("""
fig, axes = plt.subplots(1, len(PROTOCOLOS), figsize=(8 * len(PROTOCOLOS), 9), sharex=True)
for ax, protocolo in zip(axes, PROTOCOLOS):
    teste = media_todos.query("protocolo == @protocolo and parte == 'teste'")
    ordem = comparacao_bases[protocolo].index[::-1]
    posicoes = np.arange(len(ordem))
    for deslocamento, base, cor in ((-0.2, "padronizada", "tab:green"), (0.2, "original", "tab:gray")):
        valores = teste.query("base == @base").set_index("modelo").loc[ordem]
        ax.barh(posicoes + deslocamento, valores["f1_mean"], height=0.4, xerr=valores["f1_std"],
                color=cor, label=NOME_BASE[base], error_kw={"lw": 0.8})
    ax.set_yticks(posicoes)
    ax.set_yticklabels(ordem, fontsize=8)
    ax.set_xlabel("F1 de teste (média ± desvio-padrão das 15 folds)")
    ax.set_title(f"Padronizada x não padronizada — {NOME_PROTOCOLO[protocolo]}")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(out("ex033_padronizada_x_original.png"), dpi=150, bbox_inches="tight")
plt.show()
""")
code("""
# Resumo: melhor individual e melhor ensemble de cada cenário (F1 de teste, média ± desvio-padrão das 15 folds)
linhas = []
for base, protocolo in CENARIOS:
    teste = media_todos.query("base == @base and protocolo == @protocolo and parte == 'teste'").set_index("modelo")
    individual = teste.drop(U.ENSEMBLES_033, errors="ignore")["f1_mean"].idxmax()
    ensemble = teste.loc[[m for m in U.ENSEMBLES_033 if m in teste.index], "f1_mean"].idxmax()
    linhas.append({"base": NOME_BASE[base], "protocolo": NOME_PROTOCOLO[protocolo],
                   "melhor individual": individual,
                   "F1 do melhor individual": teste.loc[individual, "f1_mean"],
                   "F1 do melhor individual dp": teste.loc[individual, "f1_std"],
                   "melhor ensemble": ensemble,
                   "F1 do melhor ensemble": teste.loc[ensemble, "f1_mean"],
                   "F1 do melhor ensemble dp": teste.loc[ensemble, "f1_std"]})
resumo_cenarios = pd.DataFrame(linhas)
resumo_cenarios.to_csv(out("ex033_resumo_cenarios.csv"), index=False)
display(U.formatar_dp(resumo_cenarios))
""")

# ---------------------------------------------------------------------------- 12
md("""
## 12. External validation

The external set (1,557 compounds, 35 positives) enters no fit and no decision in this study. It is read **once**, without k-fold, by models refit on the whole training set:

- **Models:** the best individual and the best ensemble under the scaffold split (highest mean test F1), in each dataset. The final models follow section 3.1: `C`, greedy members, weights and meta-model come from a 5-fold scaffold CV over the full training set. The unstandardized dataset gets its own final models here, fitted the same way.
- **Two versions of the external set:**
  - *unstandardized*: the original SMILES, scored by the unstandardized models;
  - *standardized*: the SMILES go through the same standardization as section 1 (forms of one substance become a single row, positive if any form is positive), with embeddings generated for them, and are scored by the standardized models.
- **Overlap removed:** external compounds that are the same substance (same standardized InChIKey) as a training compound leave the evaluation, because the models have seen them.
- **Cuts:** *full* (everything but the overlap) and *novel scaffold* (only compounds whose Bemis–Murcko skeleton is absent from the training set).
- **Screening metrics**, all threshold-free: Precision@K and Recall@K for K = 50, 100 and 200, and BEDROC (α = 20). The external prevalence (≈2%) is far below the training one (17%), so these metrics compare the ordering of the compounds rather than a cut at 0.5.

The predictions are cached as a CSV. Re-running loads that file and does not read the external set again; every new read is appended to the external-read log.
""")
code("""
BASE_EXTERNA = os.path.join(ROOT, "dados", "Bases de dados", "Treinamento - limpas",
                            "Base_GeroLLM_reconstruida", "external_test_set.csv")
EXTERNA_PADRONIZADA = os.path.join(DIR_BASE, "externo_padronizado.csv")
DIR_EMBEDDINGS_EXTERNO = {
    "original": os.path.join(ROOT, "dados", "Embeddings_salvos", "Embeddings_GeroLLM_testeExterno"),
    "padronizada": os.path.join(ROOT, "dados", "Embeddings_salvos", "Embeddings_GeroLLM_testeExterno_padronizada"),
}
MODELOS_FINAIS_ORIGINAL = os.path.join(RESULTS_DIR, "modelos", "ex032_modelos_finais_original.joblib")
PREDICOES_EXTERNO = out("ex032_externo_predicoes.csv")   # mesma leitura do EX_032: os modelos finais não mudaram
LOG_EXTERNO = os.path.join(RESULTS_DIR, "json", "external_read_log_ex026.json")

# Versões do externo e a química de cada uma (InChIKey e grupo de scaffold)
externo = {"original": pd.read_csv(BASE_EXTERNA)}
if os.path.exists(EXTERNA_PADRONIZADA):
    externo["padronizada"] = pd.read_csv(EXTERNA_PADRONIZADA)
else:
    externo["padronizada"], _, _ = U.padronizar_base(externo["original"])
    externo["padronizada"].to_csv(EXTERNA_PADRONIZADA, index=False)
quimica_treino = {"original": quimica_original, "padronizada": quimica}
quimica_externo = {b: U.chemistry_table(externo[b]["smiles"].astype(str).tolist()) for b in BASES}

cortes = {b: U.external_cuts(quimica_treino[b], quimica_externo[b]) for b in BASES}
display(pd.DataFrame({NOME_BASE[b]: {
    "compostos no externo": len(externo[b]),
    "positivos no externo": int(externo[b]["label"].sum()),
    "mesma substância do treino (removidos)": cortes[b]["n_same_substance_removed"],
    "corte completo": int(cortes[b]["full"].sum()),
    "positivos no corte completo": int(externo[b]["label"].to_numpy()[cortes[b]["full"]].sum()),
    "corte scaffold inédito": int(cortes[b]["novel_scaffold"].sum()),
    "positivos no corte scaffold inédito": int(externo[b]["label"].to_numpy()[cortes[b]["novel_scaffold"]].sum()),
} for b in BASES}))
""")
code("""
# Modelos finais da base não padronizada (a padronizada já tem os seus, seção 3.1)
if os.path.exists(MODELOS_FINAIS_ORIGINAL):
    finais_original = joblib.load(MODELOS_FINAIS_ORIGINAL)
    print(f"[checkpoint] modelos finais carregados de {os.path.basename(MODELOS_FINAIS_ORIGINAL)}")
else:
    finais_original = U.treinar_finais(X["original"], Y["original"], NOMES, BINARIOS, C_GRID,
                                       GRUPOS["original"], n_jobs=N_JOBS)
    joblib.dump(finais_original, MODELOS_FINAIS_ORIGINAL)
modelos_finais = {"padronizada": finais, "original": finais_original}

# Melhor individual e melhor ensemble do split por scaffold, em cada base
avaliados = {}
for base in BASES:
    teste = media_todos.query("base == @base and protocolo == 'scaffold' and parte == 'teste'").set_index("modelo")["f1_mean"]
    avaliados[base] = [melhor_individual[base], teste[[m for m in U.ENSEMBLES if m in teste.index]].idxmax()]
    print(f"{NOME_BASE[base]}: {avaliados[base]}; membros da seleção gulosa final: "
          f"{modelos_finais[base]['membros_gulosos']}")
""")
code("""
# Embeddings do externo: só as representações que os modelos avaliados usam
necessarios = {b: sorted({n for m in avaliados[b] for n in U.encoders_do_modelo(modelos_finais[b], m)}) for b in BASES}


def arquivo_externo(base, chave):
    if base == "original" and chave == "ecfp4":      # nome do EX_027
        return os.path.join(DIR_EMBEDDINGS_EXTERNO[base], "fingerprints.npy")
    return os.path.join(DIR_EMBEDDINGS_EXTERNO[base], f"Embeddings_{chave}.npy")


def externo_pronto(base, nome):
    arquivo = arquivo_externo(base, ENCODERS[nome])
    return os.path.exists(arquivo) and np.load(arquivo, mmap_mode="r").shape[0] == len(externo[base])


faltando = [n for n in necessarios["padronizada"] if not externo_pronto("padronizada", n)]
if faltando:
    comando = [PYTHON_EMBEDDINGS, SCRIPT_EMBEDDINGS, "--csv", EXTERNA_PADRONIZADA, "--coluna", "smiles",
               "--rotulo", "label", "--saida", DIR_EMBEDDINGS_EXTERNO["padronizada"], "--encoders", *faltando]
    print("Gerando embeddings do externo padronizado:", ", ".join(faltando))
    processo = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for linha in processo.stdout:
        if linha.startswith(("[ok]", "[erro]", "[checkpoint]", "Resumo")):
            print(linha, end="")
    processo.wait()

X_externo = {}
for base in BASES:
    ausentes = [n for n in necessarios[base] if not externo_pronto(base, n)]
    assert not ausentes, f"embeddings do externo faltando ({base}): {ausentes}"
    X_externo[base] = {n: np.load(arquivo_externo(base, ENCODERS[n])).astype(np.float32) for n in necessarios[base]}
    for n in necessarios[base]:        # embeddings na mesma ordem e com os mesmos rótulos do CSV
        pasta, nome_arquivo = os.path.split(arquivo_externo(base, ENCODERS[n]))
        nome_rotulos = ("rotulos_fingerprints.npy" if nome_arquivo == "fingerprints.npy"
                        else nome_arquivo.replace("Embeddings_", "rotulos_"))
        rotulos = np.load(os.path.join(pasta, nome_rotulos)).astype(int)
        assert (rotulos == externo[base]["label"].to_numpy()).all(), (base, n)
    print(f"{NOME_BASE[base]}: {len(necessarios[base])} representações do externo carregadas")
""")
code("""
# Leitura única do externo: as predições ficam salvas e não são refeitas
if os.path.exists(PREDICOES_EXTERNO):
    predicoes_externo = pd.read_csv(PREDICOES_EXTERNO)
    print(f"[checkpoint] predições do externo carregadas de {os.path.basename(PREDICOES_EXTERNO)} (sem nova leitura)")
else:
    partes = []
    for base in BASES:
        parte = pd.DataFrame({"base": base, "linha": np.arange(len(externo[base])),
                              "label": externo[base]["label"].astype(int).to_numpy(),
                              "corte_completo": cortes[base]["full"],
                              "corte_scaffold_inedito": cortes[base]["novel_scaffold"]})
        for modelo in avaliados[base]:
            parte[modelo] = U.prever_modelo(modelos_finais[base], X_externo[base], modelo)
        partes.append(parte)
    predicoes_externo = pd.concat(partes, ignore_index=True)
    predicoes_externo.to_csv(PREDICOES_EXTERNO, index=False)
    U.log_external_read(LOG_EXTERNO, "ex032-modelos-finais",
                        note="EX_032: melhor individual e melhor ensemble do split por scaffold, "
                             "modelos finais das bases padronizada e não padronizada; externo nas "
                             "versões padronizada e não padronizada; leitura única, métricas de triagem")

linhas = []
for base in BASES:
    parte = predicoes_externo.query("base == @base")
    for corte, coluna in (("completo", "corte_completo"), ("scaffold inédito", "corte_scaffold_inedito")):
        selecao = parte[parte[coluna]]
        for modelo in avaliados[base]:
            linhas.append({"externo": NOME_BASE[base].replace("base", "externo"), "corte": corte,
                           "modelo": modelo, **U.metricas_triagem(selecao["label"].to_numpy(), selecao[modelo].to_numpy())})
externo_triagem = pd.DataFrame(linhas)
externo_triagem.to_csv(out("ex033_externo_triagem.csv"), index=False)
print("Conjunto externo, leitura única (sem k-fold)")
display(externo_triagem.round(4))
""")

# ---------------------------------------------------------------------------- 13
md("""
## 13. Confidence intervals and paired tests (limitation i)

Everything so far was reported as mean ± standard deviation over the 15 folds. That standard deviation describes **partition variability**, not the uncertainty of the estimate: the folds share training data and are not independent, so it is not a confidence interval and it cannot support a claim of significance. This section adds the two missing pieces:

- **Scaffold bootstrap.** Within one repetition (seed 42) the 5 folds cover the whole dataset and every compound is predicted exactly once, by a model that has not seen it. The bootstrap resamples **whole scaffold groups** rather than compounds: compounds sharing a skeleton are not independent observations, and resampling row by row treats a congeneric series as several independent draws, which narrows the interval without adding information. 2,000 replicates.
- **Paired bootstrap.** Two overlapping intervals do **not** imply that the difference is null when the models are correlated — and models scored on the same compounds always are. "Does A beat B" is answered by the interval of the difference, with the same resample applied to both score vectors.
- **Corrected resampled t-test (Nadeau–Bengio).** Over the 15 folds, for the claims that exist only as per-fold values. The correction replaces the 1/n variance factor by 1/n + n_test/n_train = 0.3167. The uncorrected p-value is printed alongside, to show the size of the difference.

The setting is the strictest one: **standardized dataset, scaffold split**. The reference is the best individual representation.
""")
code("""
N_BOOT = 2000
MODELOS_IC = U.ENSEMBLES_033 + [melhor_individual["padronizada"]]
cenario_ic = ("padronizada", "scaffold")

proba_oof = U.oof_de_uma_repeticao(
    Y["padronizada"], indices[cenario_ic],
    {**membros[cenario_ic], **ensembles[cenario_ic]}, MODELOS_IC, seed=42)

intervalos = U.tabela_intervalos(Y["padronizada"], proba_oof, GRUPOS["padronizada"],
                                 MODELOS_IC, n_boot=N_BOOT)
intervalos.to_csv(out("ex033_intervalos_scaffold_padronizada.csv"), index=False)
largo = intervalos.assign(
    valor=lambda d: d.apply(lambda r: f"{r['point']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]", axis=1)
).pivot(index="modelo", columns="metric", values="valor")
print(f"IC95% por bootstrap de scaffold ({N_BOOT} réplicas), base padronizada, scaffold, seed 42")
display(largo)
""")
code("""
# Cada ensemble contra o melhor individual, com o mesmo reamostramento nos dois
pareado = U.tabela_pareada(Y["padronizada"], proba_oof, GRUPOS["padronizada"],
                           U.ENSEMBLES_033, melhor_individual["padronizada"], n_boot=N_BOOT)
pareado.to_csv(out("ex033_pareado_scaffold_padronizada.csv"), index=False)
resumo = pareado.assign(
    valor=lambda d: d.apply(lambda r: f"{r['difference']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                                      + ("*" if r["excludes_zero"] else ""), axis=1)
).pivot(index="modelo", columns="metric", values="valor")
print(f"Ensemble − {melhor_individual['padronizada']}, bootstrap pareado por scaffold. "
      f"* = o intervalo exclui zero.")
display(resumo)
""")
code("""
# Nadeau-Bengio sobre as 15 folds, nos quatro cenários
por_fold = pd.concat([metricas_individuais, metricas_ensembles], ignore_index=True)
linhas = []
for base, protocolo in CENARIOS:
    parte = por_fold.query("base == @base and protocolo == @protocolo")
    tabela = U.tabela_nadeau_bengio(parte, U.ENSEMBLES_033, melhor_individual[base])
    linhas.append(tabela.assign(base=NOME_BASE[base], protocolo=NOME_PROTOCOLO[protocolo]))
nadeau = pd.concat(linhas, ignore_index=True)
nadeau.to_csv(out("ex033_nadeau_bengio.csv"), index=False)
print("Teste t corrigido de Nadeau-Bengio sobre as 15 folds (p ingênuo entre parênteses)")
display(nadeau.assign(
    p=lambda d: d.apply(lambda r: f"{r['p corrigido']:.3f} ({r['p ingênuo']:.3f})", axis=1)
).pivot_table(index=["base", "protocolo", "modelo"], columns="metrica", values="p", aggfunc="first"))
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
cenario_ic = ("padronizada", "scaffold")
alvos = ["Stacking", "Stacking (isotônica)", "Stacking (Platt)",
         "Seleção gulosa", "Seleção gulosa (PR-AUC)"]
tabela = media_ensembles.query("base == 'padronizada' and protocolo == 'scaffold' and parte == 'teste'").set_index("modelo")
display(pd.DataFrame({U.NOMES_METRICAS[m]: U.media_dp(tabela.loc[alvos, f"{m}_mean"],
                                                      tabela.loc[alvos, f"{m}_std"])
                      for m in U.METRICAS}))

# Fração de compostos classificados como positivos no limiar 0,5: o sintoma da descalibração
linhas = []
for nome in alvos:
    fracoes = [( ensembles[cenario_ic][(s, k, nome)]["test_proba"] >= U.LIMIAR_PADRAO).mean()
               for (s, k) in indices[cenario_ic]]
    linhas.append({"modelo": nome, "fração prevista positiva (teste)": np.mean(fracoes),
                   "fração prevista positiva dp": np.std(fracoes, ddof=1)})
fracao = pd.DataFrame(linhas)
fracao.to_csv(out("ex033_calibracao_scaffold_padronizada.csv"), index=False)
print(f"Prevalência real: {Y['padronizada'].mean():.2%}")
display(U.formatar_dp(fracao))
""")
code("""
# Limiar escolhido dentro do fold, grade larga: é isso que resolve a descalibração
por_fold_limiar = U.limiar_in_fold(Y["padronizada"], indices[cenario_ic],
                                   ensembles[cenario_ic], alvos)
por_fold_limiar.to_csv(out("ex033_limiar_in_fold_scaffold_padronizada.csv"), index=False)

agregado = por_fold_limiar.groupby("modelo").agg(["mean", "std"])
colunas = ["limiar escolhido"] + [f"{m}: {quando}" for m in ("f1", "precision", "recall")
                                  for quando in ("0,5", "limiar do fold")]
tabela = pd.DataFrame({c: U.media_dp(agregado[(c, "mean")], agregado[(c, "std")])
                       for c in colunas}).loc[alvos]
print("Limiar 0,5 contra o limiar escolhido no fold (média ± desvio-padrão das 15 folds)")
display(tabela)
print("Ganho de F1 ao escolher o limiar no fold:")
display((por_fold_limiar.assign(ganho=lambda d: d["f1: limiar do fold"] - d["f1: 0,5"])
         .groupby("modelo")["ganho"].agg(["mean", "std"]).loc[alvos].round(4)))
""")
code("""
# Membros escolhidos por cada critério, nas 15 folds
for base in BASES:
    cenario = (base, "scaffold")
    freq = pd.concat({
        "F1 em 0,5": U.frequencia_selecao(ensembles[cenario], NOMES, "Seleção gulosa")["frequencia"],
        "PR-AUC": U.frequencia_selecao(ensembles[cenario], NOMES, "Seleção gulosa (PR-AUC)")["frequencia"],
    }, axis=1)
    freq.to_csv(out(f"ex033_frequencia_gulosa_criterio_{base}.csv"))
    tamanhos = {c: [len(r["membros"]) for (s, k, m), r in ensembles[cenario].items() if m == c]
                for c in ("Seleção gulosa", "Seleção gulosa (PR-AUC)")}
    print(f"{NOME_BASE[base]}, scaffold — tamanho mediano: "
          + "; ".join(f"{c} {np.median(v):.0f} (de {min(v)} a {max(v)})" for c, v in tamanhos.items()))
    display(freq.sort_values("PR-AUC", ascending=False).round(3))
""")
code("""
# O estudo do limiar refeito para a seleção gulosa por PR-AUC, que não tem o pico forçado em 0,5
linhas = []
fig, axes = plt.subplots(1, len(BASES), figsize=(8 * len(BASES), 4.5), sharey=True)
for ax, base in zip(axes, BASES):
    cenario = (base, "scaffold")
    for modelo, cor in (("Seleção gulosa", "tab:red"), ("Seleção gulosa (PR-AUC)", "tab:green")):
        probas = {(s, k): r for (s, k, n), r in ensembles[cenario].items() if n == modelo}
        curva, limiar = U.estudo_limiar(Y[base], indices[cenario], probas)[1:]
        ax.plot(curva.index, curva["f1_oof_mean"], ls="-", color=cor, label=f"{modelo} — OOF")
        ax.plot(curva.index, curva["f1_teste_mean"], ls="--", color=cor,
                label=f"{modelo} — teste (só visualização)")
        ax.scatter([limiar], [curva.loc[limiar, "f1_oof_mean"]], s=160, marker="*", color=cor,
                   edgecolors="black", linewidths=0.6, zorder=4)
        linhas.append({"base": NOME_BASE[base], "critério": modelo,
                       "limiar escolhido (pico OOF)": limiar,
                       "F1 teste em 0,5": curva.loc[0.5, "f1_teste_mean"],
                       "F1 teste no limiar escolhido": curva.loc[limiar, "f1_teste_mean"]})
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("limiar")
    ax.set_title(f"Limiar por critério de seleção — {NOME_BASE[base]}, scaffold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("F1 médio (15 folds)")
fig.tight_layout()
fig.savefig(out("ex033_limiar_por_criterio_scaffold.png"), dpi=150, bbox_inches="tight")
plt.show()

limiar_criterio = pd.DataFrame(linhas)
limiar_criterio.to_csv(out("ex033_limiar_por_criterio.csv"), index=False)
display(limiar_criterio.round(4))
""")

# ---------------------------------------------------------------------------- 15
md("""
## 15. InChIKey collapse (limitation iii)

Under the scaffold split, duplicate leakage is already almost zero (0.4 compounds per fold), and the standardized dataset still loses about 0.016 F1 to the unstandardized one. The most plausible explanation is **counting**: in the unstandardized dataset a substance present in several forms is scored several times in the same test fold, and if those substances are the most studied ones, and therefore the easier ones, F1 rises without the model being any better.

The test is direct and refits nothing: take the test predictions of the **unstandardized** dataset and aggregate them by standardized InChIKey, with one prediction per substance (the mean of the probabilities) and the label by the same "positive if any form is positive" rule used by standardization. If the Δ against the standardized dataset shrinks, the drop was a counting effect. If it holds, the cause is elsewhere.
""")
code("""
chaves_original = quimica_original["L5_inchikey"].to_numpy()
MODELOS_COLAPSO = U.ENSEMBLES_033 + [melhor_individual["original"], melhor_individual["padronizada"]]
MODELOS_COLAPSO = list(dict.fromkeys(MODELOS_COLAPSO))

linhas = []
for seed in SEEDS:
    proba_orig = U.oof_de_uma_repeticao(
        Y["original"], indices[("original", "scaffold")],
        {**membros[("original", "scaffold")], **ensembles[("original", "scaffold")]},
        MODELOS_COLAPSO, seed=seed)
    y_col, proba_col = U.colapsar_por_inchikey(Y["original"], proba_orig, chaves_original)
    proba_pad = U.oof_de_uma_repeticao(
        Y["padronizada"], indices[("padronizada", "scaffold")],
        {**membros[("padronizada", "scaffold")], **ensembles[("padronizada", "scaffold")]},
        MODELOS_COLAPSO, seed=seed)
    for modelo in MODELOS_COLAPSO:
        for cenario, y_c, p_c in (("não padronizada (linhas)", Y["original"], proba_orig[modelo]),
                                  ("não padronizada (colapsada)", y_col, proba_col[modelo]),
                                  ("padronizada", Y["padronizada"], proba_pad[modelo])):
            linhas.append({"seed": seed, "modelo": modelo, "avaliação": cenario,
                           "compostos": len(y_c), **U.metricas(y_c, p_c)})
colapso = pd.DataFrame(linhas)
colapso.to_csv(out("ex033_colapso_inchikey_por_seed.csv"), index=False)

resumo_colapso = colapso.groupby(["modelo", "avaliação"])[["compostos"] + U.METRICAS].agg(["mean", "std"])
tabela = pd.DataFrame({c: U.media_dp(resumo_colapso[(c, "mean")], resumo_colapso[(c, "std")])
                       for c in ["f1", "roc_auc", "average_precision"]})
tabela.columns = [U.NOMES_METRICAS[c] for c in ["f1", "roc_auc", "average_precision"]]
tabela.insert(0, "compostos", resumo_colapso[("compostos", "mean")].round(0).astype(int))
print("Predições OOF do split por scaffold, média ± desvio-padrão das 3 repetições")
display(tabela)
""")
code("""
# O Δ padronizada − não padronizada, antes e depois do colapso
largo = colapso.pivot_table(index=["modelo", "seed"], columns="avaliação", values="f1")
deltas = pd.DataFrame({
    "Δ F1 (padronizada − linhas)": largo["padronizada"] - largo["não padronizada (linhas)"],
    "Δ F1 (padronizada − colapsada)": largo["padronizada"] - largo["não padronizada (colapsada)"],
}).groupby("modelo").agg(["mean", "std"])
tabela = pd.DataFrame({c: U.media_dp(deltas[(c, "mean")], deltas[(c, "std")])
                       for c in ["Δ F1 (padronizada − linhas)", "Δ F1 (padronizada − colapsada)"]})
tabela.to_csv(out("ex033_colapso_inchikey_delta.csv"))
print("Se a segunda coluna estiver perto de zero, a queda da padronização era efeito de contagem.")
display(tabela)
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
checkpoints_ablacao = U.Checkpoints(os.path.join(RESULTS_DIR, "checkpoints", "checkpoints_ex033_ablacao"))
ablacao = {}
for politica in U.POLITICAS_CONFLITO:
    base_pol, mantidas = U.aplicar_politica_conflito(padronizada, politica)
    y_pol = base_pol["label"].astype(int).to_numpy()
    grupos_pol = GRUPOS["padronizada"][mantidas]
    X_pol = {n: X["padronizada"][n][mantidas] for n in NOMES}
    folds_pol = U.repeated_scaffold_folds(y_pol, grupos_pol, n_splits=N_SPLITS, seeds=SEEDS)
    indices_pol = {(s, k): (tr, te) for s, k, tr, te in folds_pol}
    print(f"[{politica}] {len(y_pol)} compostos, {y_pol.sum()} positivos ({y_pol.mean():.2%})")
    # A política "positivo" é a do EX_032: mesmos rótulos, mesmas folds, mesmos checkpoints.
    # Só as outras duas precisam de ajustes novos (2 x 18 x 15 = 540).
    reaproveita = politica == "positivo"
    membros_pol = U.rodar_membros(
        X_pol, y_pol, folds_pol, NOMES, BINARIOS, C_GRID,
        checkpoints if reaproveita else checkpoints_ablacao, "scaffold", grupos=grupos_pol,
        n_jobs=N_JOBS, prefixo="padronizada_scaffold" if reaproveita else f"ablacao_{politica}")
    ensembles_pol = {}
    for (seed, k), (tr, te) in indices_pol.items():
        registros = {n: membros_pol[(seed, k, n)] for n in NOMES}
        for nome, registro in U.ensembles_do_fold(y_pol, tr, registros, NOMES, seed).items():
            ensembles_pol[(seed, k, nome)] = registro
    ablacao[politica] = {"y": y_pol, "indices": indices_pol,
                         "metricas": pd.concat([
                             U.tabela_metricas(y_pol, indices_pol, membros_pol, "scaffold", politica),
                             U.tabela_metricas(y_pol, indices_pol, ensembles_pol, "scaffold", politica)],
                             ignore_index=True)}

metricas_ablacao = pd.concat([ablacao[p]["metricas"] for p in U.POLITICAS_CONFLITO], ignore_index=True)
metricas_ablacao.to_csv(out("ex033_ablacao_conflitos_por_fold.csv"), index=False)
media_ablacao = U.media_por_modelo(metricas_ablacao)
media_ablacao.to_csv(out("ex033_ablacao_conflitos_media.csv"), index=False)
""")
code("""
# A ordem das representações muda com a política?
teste_ablacao = media_ablacao.query("parte == 'teste'")
ordens = {p: teste_ablacao.query("base == @p").set_index("modelo")["f1_mean"] for p in U.POLITICAS_CONFLITO}
linhas = []
for metrica in ("f1", "roc_auc", "average_precision"):
    valores = {p: teste_ablacao.query("base == @p").set_index("modelo")[f"{metrica}_mean"]
               for p in U.POLITICAS_CONFLITO}
    comum = sorted(set.intersection(*(set(v.index) for v in valores.values())))
    for a, b in (("positivo", "removido"), ("positivo", "negativo")):
        rho = spearmanr(valores[a].loc[comum], valores[b].loc[comum]).statistic
        linhas.append({"métrica": U.NOMES_METRICAS[metrica], "comparação": f"{a} x {b}",
                       "Spearman": rho})
spearman_ablacao = pd.DataFrame(linhas).pivot(index="métrica", columns="comparação", values="Spearman")
spearman_ablacao.to_csv(out("ex033_ablacao_spearman.csv"))
print(f"Correlação de Spearman entre os rankings dos {len(comum)} modelos sob cada política")
display(spearman_ablacao.round(3))
""")
code("""
# A vantagem do ensemble sobrevive a cada política?
linhas = []
for politica in U.POLITICAS_CONFLITO:
    t = teste_ablacao.query("base == @politica").set_index("modelo")
    individuais = t.drop(U.ENSEMBLES_033, errors="ignore")
    melhor_ind = individuais["f1_mean"].idxmax()
    for metrica in ("f1", "roc_auc", "average_precision", "mcc"):
        melhor_ens = t.loc[[m for m in U.ENSEMBLES_033 if m in t.index], f"{metrica}_mean"].idxmax()
        linhas.append({"política": politica, "métrica": U.NOMES_METRICAS[metrica],
                       "melhor individual": melhor_ind,
                       "melhor ensemble": melhor_ens,
                       "Δ (ensemble − individual)": t.loc[melhor_ens, f"{metrica}_mean"]
                                                    - individuais[f"{metrica}_mean"].max()})
ganho_ablacao = pd.DataFrame(linhas)
ganho_ablacao.to_csv(out("ex033_ablacao_ganho_ensemble.csv"), index=False)
print("Se o Δ for positivo nas três políticas, a vantagem do ensemble não depende da regra de rótulo.")
display(ganho_ablacao.pivot(index="política", columns="métrica", values="Δ (ensemble − individual)").round(4))
display(ganho_ablacao)
""")

# =========================================================================================
for posicao, cell in enumerate(cells):
    cell["id"] = f"ex033-{posicao:03d}"
    cell["source"] = cell["source"].splitlines(keepends=True)

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3 (PIBIC)", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python", "version": "3.14"}},
    "nbformat": 4, "nbformat_minor": 5,
}
destino = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "notebooks", NOTEBOOK))
with open(destino, "w") as arquivo:
    json.dump(notebook, arquivo, indent=1, ensure_ascii=False)
print(f"escrito {destino} ({len(cells)} células)")
