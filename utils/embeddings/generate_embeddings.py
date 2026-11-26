"""
Computes the embeddings of the available encoders for a SMILES dataset, one .npy per encoder.

Each encoder is a checkpoint: if Embeddings_<key>.npy already exists with the right number of
rows it is skipped, unless --force is given. An interrupted run therefore resumes where it
stopped, and the notebook decides whether to use the files on disk or recompute everything.

Uses the functions in Embeddings.py. Runs in the `embeddings` conda environment (torch,
transformers, torch_geometric, gensim, selfies, unimol_tools); MAT and R-MAT additionally need
the huggingmolecules package (github.com/gmum/huggingmolecules), which is not on PyPI.

Usage:
    python utils/embeddings/generate_embeddings.py \\
        --csv primary_standardized.csv --column smiles --out folder/ \\
        [--encoders MolFormer-XL ...] [--force]

Outputs in --out:
    Embeddings_<key>.npy   -> matrix (n_molecules, dim) per encoder, in the order of the CSV
    labels_<key>.npy      -> labels (the --label column), in the same order
    erros_<key>.txt        -> molecules the encoder could not represent, if any
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import Embeddings  # noqa: E402

# ChemGPT: the 4.7M-parameter variant, whose embedding has 128 dimensions.
Embeddings.CHEMGPT_ID = "ncfrey/ChemGPT-4.7M"

# name in the study -> (file key, per-molecule function, batch function)
ENCODERS = {
    "MolFormer-XL": ("molformer", Embeddings.embed_molformer, None),
    "SELFormer": ("selformer", Embeddings.embed_selformer, None),
    "ChemBERTa-2": ("chembert", Embeddings.embed_chemberta2, None),
    "ChemBERTa-3": ("chemberta3", Embeddings.embed_chemberta3, None),
    "MolDeBERTa": ("moldeberta", Embeddings.embed_moldeberta, None),
    "MolCLR": ("molclr", Embeddings.embed_molclr, None),
    "Mole-BERT": ("molebert", Embeddings.embed_molebert, None),
    "GIN": ("gin", Embeddings.embed_gin, None),
    "GraphMVP": ("graphmvp", Embeddings.embed_graphmvp, None),
    "Uni-Mol": ("unimol", None, Embeddings.embed_unimol_batch),
    "ECFP4-2048": ("ecfp4", Embeddings.embed_ecfp4, None),
    "Atom-Pair": ("atompair", Embeddings.embed_atompair, None),
    "Graph-FP": ("graphfp", Embeddings.embed_graphfp, None),
    "TT": ("tt", Embeddings.embed_tt, None),
    "Mol2Vec": ("mol2vec", Embeddings.embed_mol2vec, None),
    "ChemGPT": ("chemgpt", Embeddings.embed_chemgpt, None),
    "ChemFM": ("chemfm", Embeddings.embed_chemfm, None),
    "MAT": ("mat", Embeddings.embed_mat, None),
    "R-MAT": ("rmat", Embeddings.embed_rmat, None),
    "CLAMP": ("clamp", Embeddings.embed_clamp, None),
}


def path(out, chave, prefixo="Embeddings"):
    return os.path.join(out, f"{prefixo}_{chave}.npy")


def pronto(out, chave, n):
    file_path = path(out, chave)
    return os.path.exists(file_path) and np.load(file_path, mmap_mode="r").shape[0] == n


def gerar(name, smiles):
    """Embeddings of one encoder for the list of SMILES, one row per molecule."""
    _, por_molecula, em_lote = ENCODERS[name]
    if em_lote is not None:
        return np.asarray(em_lote(smiles), dtype=np.float32), []
    vectors, failures = [], []
    for i, smile in enumerate(smiles):
        try:
            vetor = por_molecula(smile)
        except Exception as error:          # one bad molecule must not abort the encoder
            vetor, reason = None, f"{type(error).__name__}: {error}"
        else:
            reason = "the encoder returned no vector"
        if vetor is None:
            failures.append((i, smile, reason))
            vectors.append(None)
        else:
            vectors.append(np.asarray(vetor, dtype=np.float32).ravel())
    dimension = next((len(v) for v in vectors if v is not None), None)
    if dimension is None:
        raise RuntimeError(f"{name}: no molecule could be represented ({failures[0][2]})")
    # A molecule without a representation becomes a row of zeros, to keep the alignment with
    # the CSV; those cases are listed in erros_<key>.txt.
    matriz = np.vstack([v if v is not None else np.zeros(dimension, np.float32) for v in vectors])
    return matriz, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--csv", required=True)
    parser.add_argument("--column", default="smiles")
    parser.add_argument("--label", default="label")
    parser.add_argument("--out", required=True)
    parser.add_argument("--encoders", nargs="*", default=list(ENCODERS))
    parser.add_argument("--force", action="store_true", help="recompute even if the file exists")
    args = parser.parse_args()

    base = pd.read_csv(args.csv)
    smiles = base[args.column].astype(str).tolist()
    rotulos = base[args.label].to_numpy()
    os.makedirs(args.out, exist_ok=True)

    status = {}
    for name in args.encoders:
        chave = ENCODERS[name][0]
        if not args.force and pronto(args.out, chave, len(smiles)):
            print(f"[checkpoint] {name}: already on disk, skipping", flush=True)
            status[name] = "existing"
            continue
        inicio = time.time()
        try:
            matriz, failures = gerar(name, smiles)
        except Exception as error:
            print(f"[error] {name}: {type(error).__name__}: {error}", flush=True)
            status[name] = f"error: {type(error).__name__}"
            continue
        np.save(path(args.out, chave), matriz)
        np.save(path(args.out, chave, "labels"), rotulos)
        if failures:
            with open(os.path.join(args.out, f"erros_{chave}.txt"), "w") as file_path:
                for i, smile, reason in failures:
                    file_path.write(f"{i}\t{smile}\t{reason}\n")
        print(f"[ok] {name}: {matriz.shape} em {time.time() - inicio:.0f}s"
              + (f", {len(failures)} molecules without a vector" if failures else ""), flush=True)
        status[name] = "generated"

    faltando = [n for n, s in status.items() if s.startswith("error")]
    print(f"Summary: {sum(s == 'generated' for s in status.values())} generated, "
          f"{sum(s == 'existing' for s in status.values())} already on disk, "
          f"{len(faltando)} with errors {faltando if faltando else ''}")
    sys.exit(1 if faltando else 0)


if __name__ == "__main__":
    main()
