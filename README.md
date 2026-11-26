# Geroprotector screening with frozen molecular encoders
Runnable code, input data and stored predictions for a study on ranking candidate
geroprotectors with pretrained molecular encoders. It accompanies a manuscript currently under
review; the manuscript itself is **not** part of this repository, and the text below was
written for the repository rather than copied from it.

What the experiments do: 18 pretrained encoders are used frozen, each feeding a single L2
logistic regression; seven ensembles are built on top of them; and every model is scored in
four settings that cross two ways of deduplicating the dataset (merging by standardized
InChIKey, or by canonical SMILES only) with two ways of splitting it (grouped by Bemis–Murcko
skeleton, or stratified at random), each repeated over 5 folds × 3 seeds.

Everything is produced by one notebook:
[`notebooks/benchmark_four_protocols.ipynb`](notebooks/benchmark_four_protocols.ipynb).

## Two ways to reproduce

### 1. Check every reported number in seconds

```bash
python utils/logistic_regression/paper_numbers.py
```

Reads the per-fold metrics in `results/tables/` and prints each number reported in the
manuscript next to the value computed from those files, with a verdict. No GPU, no embeddings,
no model fitting. The output of our run is in
[`results/paper_numbers.txt`](results/paper_numbers.txt): **100 of 100 checked numbers match**.

### 2. Re-run the benchmark

```bash
conda env create -f environment.yml && conda activate PIBIC_Luiggy
jupyter lab notebooks/benchmark_four_protocols.ipynb
```

The 1,080 member fits (2 datasets × 2 splits × 15 folds × 18 encoders, each with its own inner
loop) are **not** recomputed: `results/checkpoints/member_fits/` holds, for every fit, the
selected `C`, the inner out-of-fold probability column and the outer test column. Every
ensemble, calibrator, threshold, bootstrap interval and external-set number is derived from
those stored predictions. `results/checkpoints/label_ablation/` holds the 540 extra fits of the
label ablation, which cannot reuse them because changing labels invalidates them.

What the repository does **not** ship is the 18 embedding matrices (866 MB of `.npy`), which
section 2 of the notebook loads into memory. They are derived data; regenerate them with

```bash
python utils/embeddings/generate_embeddings.py \
    --csv data/gerollm_standardized/primary_standardized.csv \
    --column smiles --label label --out data/embeddings/standardized_train
```

and likewise for the other three cuts (the unstandardized training set and both versions of the
external set). Each encoder writes `Embeddings_<key>.npy` plus `labels_<key>.npy`; if you have
matrices from an earlier run whose label files are named `rotulos_*.npy`, rename them with
`for f in rotulos_*.npy; do mv "$f" "labels_${f#rotulos_}"; done`. [`notebooks/embeddings_generation.ipynb`](notebooks/embeddings_generation.ipynb)
drives this, reports what is already on disk and audits the alignment contract. Each encoder is
a checkpoint, so an interrupted run resumes. The encoders need a separate environment (torch,
transformers, torch_geometric, gensim, selfies, unimol_tools); that notebook creates it.

## Layout

```
notebooks/
  benchmark_four_protocols.ipynb        the experiments: every reported number
  embeddings_generation.ipynb           generates and audits the 18 embedding matrices
utils/
  logistic_regression/
    core_utils.py                       standardization, folds, pipeline, bootstrap helpers
    protocol_utils.py                   the protocol: nested fits, metrics, ensembles
    analysis_utils.py                   calibration, threshold-free selection, CIs, ablation
    build_benchmark_notebook.py         generates the notebook above (it is built from this
                                        script, never edited by hand)
    paper_numbers.py                    checks the reported numbers against results/tables/
    paper_figures.py                    redraws the figures as vector PDFs
  embeddings/
    Embeddings.py                       one function per encoder
    generate_embeddings.py              the CLI the notebooks drive
data/
  gerollm_reconstructed/                primary_dataset_dedup.csv (4,233 compounds),
                                        external_test_set.csv (1,557)
  gerollm_standardized/                 primary_standardized.csv (3,752 substances),
                                        external_standardized.csv, and the standardization audit
results/
  checkpoints/                          the 1,080 member fits + 540 ablation fits
  tables/                               per-fold and averaged metrics, intervals, ablations
  figures/                              figures, with vector PDFs under figures/paper/
  models/                               final models, fitted on the full training set
```


## Executed outputs

The notebook ships without stored outputs, so that the committed file stays a clean record of
the code. Our run is in `results/`: the tables under `results/tables/`, the figures under
`results/figures/`, and the full check of the reported numbers in `results/paper_numbers.txt`.
