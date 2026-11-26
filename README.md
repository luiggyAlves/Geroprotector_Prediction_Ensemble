# Combining Frozen Molecular Representations for Geroprotector Prediction

Code, data and stored predictions behind every number in the paper: 18 pretrained molecular
representations used as frozen feature extractors, each with an L2-regularized logistic
regression, plus five ensembles, benchmarked in four paired evaluation settings (chemically
standardized / conventional dataset × Bemis–Murcko scaffold split / random stratified split,
5 folds × 3 seeds each).

One notebook produces the whole benchmark:
[`notebooks/EX_032_033_benchmark_completo.ipynb`](notebooks/EX_032_033_benchmark_completo.ipynb).
Sections 1–12 are the benchmark itself; sections 13–16 are the analyses that address five of
the eight limitations discussed in the paper (bootstrap confidence intervals, calibration and
threshold-free selection, InChIKey collapse, label-conflict ablation).

The notebook and the code comments are in Portuguese; this README, the file names and the
column names are in English.

## Two ways to reproduce

### 1. Check every number in seconds

```bash
python utils/regressao_logistica/numeros_do_artigo.py
```

Reads the stored per-fold metrics in `resultados/csv/` and prints each claim in the paper next
to the value computed from those files, with a verdict. It needs no GPU, no embeddings and no
model fitting. The output of our run is in
[`resultados/numeros_do_artigo.txt`](resultados/numeros_do_artigo.txt): **100 of 100 checked
numbers match**.

### 2. Re-run the benchmark

```bash
conda env create -f environment.yml && conda activate PIBIC_Luiggy
jupyter lab notebooks/EX_032_033_benchmark_completo.ipynb
```

The 1,080 member fits (2 datasets × 2 protocols × 15 folds × 18 representations, each with its
own inner loop) are **not** recomputed: `resultados/checkpoints/checkpoints_ex032/` holds, for
every fit, the selected `C`, the inner out-of-fold probability column and the outer test
column. Every ensemble, calibrator, threshold, bootstrap interval and external-set number is
derived from those stored predictions. `checkpoints_ex033_ablacao/` holds the 540 extra fits of
the label ablation, which cannot reuse them because changing labels invalidates them.

What the repository does **not** ship is the 18 embedding matrices (866 MB of `.npy`), which
section 2 of the notebook loads into memory. They are derived data: regenerate them with

```bash
python utils/embeddings/gerar_embeddings_base.py \
    --csv "dados/Bases de dados/Treinamento - limpas/Base_GeroLLM_padronizada/primary_padronizada.csv" \
    --coluna smiles --rotulo label --saida dados/Embeddings_salvos/Embeddings_GeroLLM_padronizada
```

and the same for the three other cuts (unstandardized training set, and both versions of the
external set). [`notebooks/geracao_embeddings.ipynb`](notebooks/geracao_embeddings.ipynb)
drives this, reports what is already on disk and checks the alignment contract. Each encoder is
a checkpoint, so an interrupted run resumes. The encoders need a separate environment (torch,
transformers, torch_geometric, gensim, selfies, unimol_tools); that notebook creates it.

## Layout

```
notebooks/
  EX_032_033_benchmark_completo.ipynb   the benchmark: every number in the paper
  geracao_embeddings.ipynb              generates and audits the 18 embedding matrices
utils/
  regressao_logistica/
    ex027_utils.py ex032_utils.py ex033_utils.py   protocol, ensembles, metrics, statistics
    build_ex032_033.py                  generates the notebook above (the notebook is built
                                        from this script, not edited by hand)
    numeros_do_artigo.py                checks the paper against resultados/csv/
    figuras_artigo_ex033.py             regenerates the paper figures
  embeddings/                           one function per encoder, plus the CLI
dados/Bases de dados/Treinamento - limpas/
  Base_GeroLLM_reconstruida/            primary_dataset_dedup.csv (4,233 compounds),
                                        external_test_set.csv (1,557)
  Base_GeroLLM_padronizada/             primary_padronizada.csv (3,752 substances),
                                        externo_padronizado.csv, and the standardization audit
resultados/
  checkpoints/                          the 1,080 + 540 stored fits
  csv/                                  per-fold and averaged metrics, intervals, ablations
  figuras/                              figures
  modelos/                              final models, fitted on the full training set
```

## One thing to know before reading the aggregate tables

Sections 7, 8 and 11 aggregate *across models*: the mean Δ per model and the Spearman
correlation between rankings. The result depends on which models enter the average, and this
study grew by three ensembles along the way — Greedy (AP), Platt-calibrated stacking and
isotonic-calibrated stacking were added after the first round. The paper quotes the
**22-model** set (18 representations + 4 ensembles): Δ F1 of +0.042 and +0.0215, Δ ROC-AUC of
+0.041 and +0.024, the −0.016 drop with 19 of 22 models declining, and the minimum Spearman of
0.948. With all 25 models the same quantities are +0.047, +0.024, +0.042, +0.025, −0.021 with
22 of 25 declining, and a minimum Spearman of 0.862.

Both are computed and printed side by side, labelled by set, and saved to
`resultados/csv/ex033_agregados_por_conjunto.csv`. The ranking tables carry both positions, so
"Graph-FP ranks 15th on test" (22 models) is 16th among all 25. Nothing else in the paper
depends on this choice: per-model metrics, ensembles, confidence intervals and the external set
are model by model.

## Data provenance

The primary dataset is reconstructed following the consolidation strategy of Gero-LLM from
public sources (AgeXtend, DrugAge, *C. elegans* screens, senolytics collections,
Geroprotectors.org). The files here are the reconstruction and its standardization audit;
redistribution of any upstream data remains subject to the original licensing terms. The
standardized dataset applies RDKit `rdMolStandardize` (cleanup, fragment parent, uncharging,
canonical tautomer) and merges rows sharing an InChIKey.

## Executed outputs

The notebook ships with the outputs of our run in the 36 cells whose code is unchanged from
that run. The four cells that compute the two model sets (sections 7, 8, 11 and the ablation
Spearman) were edited after it and carry no outputs; re-running the notebook fills them, and
`numeros_do_artigo.py` already reproduces those values from the stored metrics.
