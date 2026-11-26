"""Builds the embeddings notebook.

Writes notebooks/embeddings_generation.ipynb, the control panel over
utils/embeddings/generate_embeddings.py: it reports what is already on disk, generates what is
missing and audits what was generated.

Usage:
    python utils/embeddings/build_embeddings_notebook.py
"""

import json
import os

NOTEBOOK = "embeddings_generation.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.strip("\n")})


# =========================================================================================
md("""
# Embeddings of the 18 frozen encoders

This notebook is the control panel for the embeddings of the study: it says what is already on
disk, generates what is missing and audits what was generated. One row per molecule, in the
order of the CSV — that alignment is what the benchmark notebook relies on when it loads the
`.npy` files.

The computation itself does **not** live here. It lives in `utils/embeddings/`:

- `Embeddings.py` — one function per encoder, with the checkpoints and the featurizations;
- `generate_embeddings.py` — the CLI that walks a dataset, calls the right encoder and writes
  `Embeddings_<key>.npy` + `labels_<key>.npy` (and `erros_<key>.txt`, when a molecule cannot be
  represented).

This notebook only drives that CLI, and that is deliberate: the encoders do not fit in the
analysis environment (torch, transformers, torch_geometric, gensim, selfies, unimol_tools), so
they run in a separate conda environment, in a subprocess.

**One checkpoint per encoder:** if the `.npy` already exists with the right number of rows, it
is skipped. Running the whole notebook again recomputes nothing, and an interrupted run resumes
where it stopped.
""")

# ---------------------------------------------------------------------------- 1
md("""
## 1. Setup

The 18 encoders are the ones the benchmark uses, with the same file keys, so that the `.npy`
files are read there without any name translation.

MAT and R-MAT are left out: they depend on the `huggingmolecules` package, which only exists on
GitHub and does not install in the encoder environment. `generate_embeddings.py` still knows
them, so installing the package and adding them to the dictionary below brings them back.

Both datasets are the **standardized** ones (`rdMolStandardize`, the same chemistry as the
benchmark). The unstandardized cuts are read from their own folders and are not written here.
""")
code("""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from IPython.display import display

# This notebook runs from notebooks/.
ROOT = os.path.abspath("..")                          # the repository root
DATA_DIR = os.path.join(ROOT, "data", "gerollm_standardized")
EMBEDDINGS_ROOT = os.path.join(ROOT, "data", "embeddings")
EMBEDDINGS_SCRIPT = os.path.join(ROOT, "utils", "embeddings", "generate_embeddings.py")

ENCODERS = {  # name in the study -> key of the embedding file
    "MolFormer-XL": "molformer", "SELFormer": "selformer", "ChemBERTa-2": "chembert",
    "ChemBERTa-3": "chemberta3", "MolDeBERTa": "moldeberta", "MolCLR": "molclr",
    "Mole-BERT": "molebert", "GIN": "gin", "GraphMVP": "graphmvp", "Uni-Mol": "unimol",
    "ECFP4-2048": "ecfp4", "Atom-Pair": "atompair", "Graph-FP": "graphfp", "TT": "tt",
    "Mol2Vec": "mol2vec", "ChemGPT": "chemgpt", "ChemFM": "chemfm", "CLAMP": "clamp",
}
BINARY_FPS = {"ECFP4-2048", "Atom-Pair", "Graph-FP", "TT"}   # fingerprints, not dense embeddings

# One entry per dataset: which CSV it comes from and where its .npy files go.
DATASETS = {
    "train": {"csv": os.path.join(DATA_DIR, "primary_standardized.csv"),
              "out": os.path.join(EMBEDDINGS_ROOT, "standardized_train")},
    "external": {"csv": os.path.join(DATA_DIR, "external_standardized.csv"),
                 "out": os.path.join(EMBEDDINGS_ROOT, "standardized_external")},
}
SMILES_COLUMN = "smiles"
LABEL_COLUMN = "label"

for name, info in DATASETS.items():
    dataset = pd.read_csv(info["csv"])
    info["n"] = len(dataset)
    info["labels"] = dataset[LABEL_COLUMN].to_numpy()
    os.makedirs(info["out"], exist_ok=True)
    print(f"[{name}] {info['n']} molecules, {info['labels'].mean():.1%} positive "
          f"— {os.path.basename(info['csv'])}")
print(f"\\n{len(ENCODERS)} encoders; CLI: {os.path.relpath(EMBEDDINGS_SCRIPT, ROOT)}")
""")

# ---------------------------------------------------------------------------- 2
md("""
## 2. What is already on disk

An encoder counts as ready when the `.npy` exists **and** has one row per molecule of the CSV.
The row count is the check that matters: a file generated for an earlier version of the dataset
survives on disk under the right name with the wrong length, and would be loaded out of step
with the labels.
""")
code("""
def files_of(dataset, key):
    folder = DATASETS[dataset]["out"]
    return (os.path.join(folder, f"Embeddings_{key}.npy"),
            os.path.join(folder, f"labels_{key}.npy"))


def is_ready(dataset, name):
    path = files_of(dataset, ENCODERS[name])[0]
    return (os.path.exists(path)
            and np.load(path, mmap_mode="r").shape[0] == DATASETS[dataset]["n"])


def state(dataset, name):
    path = files_of(dataset, ENCODERS[name])[0]
    if not os.path.exists(path):
        return "absent"
    n = np.load(path, mmap_mode="r").shape[0]
    if n != DATASETS[dataset]["n"]:
        return f"misaligned ({n} rows, the dataset has {DATASETS[dataset]['n']})"
    return "ok"


inventory = pd.DataFrame({dataset: {name: state(dataset, name) for name in ENCODERS}
                          for dataset in DATASETS})
display(inventory)

MISSING = {dataset: [n for n in ENCODERS if not is_ready(dataset, n)] for dataset in DATASETS}
for dataset, names in MISSING.items():
    print(f"[{dataset}] {len(ENCODERS) - len(names)}/{len(ENCODERS)} ready"
          + (f"; to generate: {', '.join(names)}" if names else ""))
""")

# ---------------------------------------------------------------------------- 3
md("""
## 3. The encoder environment

The encoders need torch, transformers, torch_geometric, gensim, selfies and unimol_tools, which
do not fit in the analysis environment. They run in a separate conda environment, `embeddings`,
discovered through the `conda` on PATH and created on the spot if it does not exist, with the
packages installed.

Discovering it rather than hard-coding a path is what lets this notebook run on a machine where
the environment lives under `~/miniconda3` and on one where it lives under Homebrew. The cell
only does work if there is something to generate.
""")
code("""
ENCODER_ENV = "embeddings"
ENCODER_REQUIREMENTS = [
    "numpy>=2", "pandas", "scikit-learn", "matplotlib", "rdkit", "torch", "transformers",
    "tokenizers", "safetensors", "huggingface-hub", "torch_geometric", "gensim", "selfies",
    "unimol-tools",
]


def encoder_python():
    conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
    assert conda, "conda was not found on PATH; it is needed for the encoder environment"
    def environments():
        out = subprocess.run([conda, "env", "list", "--json"], capture_output=True,
                             text=True, check=True).stdout
        return json.loads(out)["envs"]
    prefix = next((e for e in environments() if os.path.basename(e) == ENCODER_ENV), None)
    if prefix is None:
        print(f"[env] creating the conda environment '{ENCODER_ENV}' (python 3.11)")
        subprocess.run([conda, "create", "-y", "-q", "-n", ENCODER_ENV, "python=3.11"],
                       check=True)
        prefix = next(e for e in environments() if os.path.basename(e) == ENCODER_ENV)
    python = os.path.join(prefix, "bin", "python")
    print(f"[env] installing/checking the encoder packages in {python}")
    subprocess.run([python, "-m", "pip", "install", "-q", *ENCODER_REQUIREMENTS], check=True)
    return python


ENCODER_PYTHON = encoder_python() if any(MISSING.values()) else None
print("Nothing to generate; the environment was left untouched." if ENCODER_PYTHON is None
      else "[env] ready")
""")

# ---------------------------------------------------------------------------- 4
md("""
## 4. Generation

One CLI call per dataset, with only the encoders that are missing. The output is filtered down
to the lines that matter (`[ok]`, `[error]`, `[checkpoint]`, `Summary`), because the progress of
the larger encoders floods the output with download bars.

To redo an encoder on purpose — a new checkpoint, a corrected dataset — name it in `REGENERATE`:
it then enters the call even if it is ready, with `--force`.
""")
code("""
REGENERATE = []      # e.g. ["Uni-Mol", "CLAMP"]; [] = generate only what is missing

for dataset, names in MISSING.items():
    to_generate = names + [n for n in REGENERATE if n not in names]
    if not to_generate:
        print(f"[{dataset}] nothing to do")
        continue
    info = DATASETS[dataset]
    print(f"\\n[{dataset}] generating: {', '.join(to_generate)}")
    command = [ENCODER_PYTHON, EMBEDDINGS_SCRIPT, "--csv", info["csv"],
               "--column", SMILES_COLUMN, "--label", LABEL_COLUMN,
               "--out", info["out"], "--encoders", *to_generate]
    if REGENERATE:
        command.append("--force")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True)
    for line in process.stdout:
        if line.startswith(("[ok]", "[error]", "[checkpoint]", "Summary")):
            print(line, end="")
    process.wait()
    print(f"[{dataset}] exit code {process.returncode}")
""")

# ---------------------------------------------------------------------------- 5
md("""
## 5. Audit

Three things can be wrong with a `.npy` that exists and has the right length, and all three
would break the benchmark silently:

1. **swapped labels** — `labels_<key>.npy` must be identical to the `label` column of the CSV,
   in the same order. That is what makes it legitimate to use a single `y` vector for every `X`;
2. **molecules without a vector** — the CLI writes zeros on the row of a molecule the encoder
   could not represent, so as not to shift everything else, and lists the case in
   `erros_<key>.txt`. A row of zeros is a compound without information, not a read error;
3. **an unexpected dimension** — useful for noticing a checkpoint different from the one used
   before.
""")
code("""
audit = []
for dataset, info in DATASETS.items():
    for name, key in ENCODERS.items():
        x_path, y_path = files_of(dataset, key)
        if not os.path.exists(x_path):
            audit.append({"dataset": dataset, "encoder": name, "rows": "—"})
            continue
        X = np.load(x_path)
        y = np.load(y_path) if os.path.exists(y_path) else None
        errors = os.path.join(info["out"], f"erros_{key}.txt")
        audit.append({
            "dataset": dataset, "encoder": name, "rows": X.shape[0], "dim": X.shape[1],
            "rows == dataset": X.shape[0] == info["n"],
            "labels == CSV": (y is not None and y.shape == info["labels"].shape
                              and bool((y == info["labels"]).all())),
            "all-zero rows": int((~X.any(axis=1)).sum()),
            "listed in erros_*.txt": sum(1 for _ in open(errors)) if os.path.exists(errors) else 0,
        })

audit = pd.DataFrame(audit).set_index(["dataset", "encoder"])
display(audit)

problems = audit[(audit["rows == dataset"] != True) | (audit["labels == CSV"] != True)]
print(f"{len(audit) - len(problems)} of {len(audit)} (dataset, encoder) pairs aligned to the CSV")
if len(problems):
    print("Missing or misaligned — do not load these in the benchmark before generating them:")
    display(problems)
""")

md("""
## Notes

- **The alignment contract.** One row per row of the CSV, in the order of the CSV, including for
  a molecule that failed (a row of zeros). Nothing is filtered during generation, which is what
  makes a single `y` vector correct for every `X`; section 5 is what verifies it.
- **The four fingerprints** (ECFP4-2048, Atom-Pair, Graph-FP, TT) are binary RDKit vectors, not
  dense embeddings: they run in seconds and need no GPU. The other 14 download a checkpoint from
  the HF Hub the first time.
- **ChemGPT** uses the 4.7M-parameter variant (128 dimensions), pinned in the CLI.
- **Uni-Mol** is the only one that runs in batch, and it needs a 3D conformer; when it cannot
  generate one it falls back to 2D, which affected 1.1% of the standardized compounds.
- **Regenerating is expensive and changes nothing:** the encoders are frozen, with no
  fine-tuning. It is only worth redoing when the dataset changes or when an encoder checkpoint
  changes.
""")

# =========================================================================================
for position, cell in enumerate(cells):
    cell["id"] = f"embeddings-{position:03d}"
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
with open(destination, "w") as handle:
    json.dump(notebook, handle, indent=1, ensure_ascii=False)
print(f"written {destination} ({len(cells)} cells)")
