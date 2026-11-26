"""
Gera os embeddings dos 20 encoders para uma base de SMILES e salva um .npy por encoder.

Cada encoder é um checkpoint: se Embeddings_<chave>.npy já existe com o número certo de
linhas, ele é pulado (a não ser com --forcar). Assim uma execução interrompida continua de
onde parou, e o notebook decide se usa os arquivos prontos ou gera tudo de novo.

Usa as funções de Embeddings.py. Roda no env `embeddings` (torch, transformers,
torch_geometric, gensim, selfies, unimol_tools); MAT e R-MAT pedem também o pacote
huggingmolecules (github.com/gmum/huggingmolecules), que não está no PyPI.

Uso:
    ~/miniconda3/envs/embeddings/bin/python utils/embeddings/gerar_embeddings_base.py \\
        --csv base_padronizada.csv --coluna smiles --saida pasta/ [--encoders MolFormer-XL ...] [--forcar]

Saídas em --saida:
    Embeddings_<chave>.npy   -> matriz (n_moleculas, dim) por encoder, na ordem do CSV
    rotulos_<chave>.npy      -> rótulos (coluna --rotulo), na mesma ordem
    erros_<chave>.txt        -> moléculas que o encoder não conseguiu representar (se houver)
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import Embeddings  # noqa: E402

# ChemGPT: o EX_027 usou a variante de 4.7M parâmetros (embedding de 128 dimensões).
Embeddings.CHEMGPT_ID = "ncfrey/ChemGPT-4.7M"

# nome no estudo -> (chave do arquivo, função por molécula, função em lote)
ENCODERS = {
    "MolFormer-XL": ("molformer", Embeddings.gerar_embedding_molformer, None),
    "SELFormer": ("selformer", Embeddings.gerar_embedding_selformer, None),
    "ChemBERTa-2": ("chembert", Embeddings.gerar_embedding_chembert, None),
    "ChemBERTa-3": ("chemberta3", Embeddings.gerar_embedding_chemberta3, None),
    "MolDeBERTa": ("moldeberta", Embeddings.gerar_embedding_moldeberta, None),
    "MolCLR": ("molclr", Embeddings.gerar_embedding_molclr, None),
    "Mole-BERT": ("molebert", Embeddings.gerar_embedding_molebert, None),
    "GIN": ("gin", Embeddings.gerar_embedding_gin, None),
    "GraphMVP": ("graphmvp", Embeddings.gerar_embedding_graphmvp, None),
    "Uni-Mol": ("unimol", None, Embeddings.gerar_embeddings_unimol_batch),
    "ECFP4-2048": ("ecfp4", Embeddings.gerar_fingerprint, None),
    "Atom-Pair": ("atompair", Embeddings.gerar_embedding_atompair, None),
    "Graph-FP": ("graphfp", Embeddings.gerar_embedding_graphfp, None),
    "TT": ("tt", Embeddings.gerar_embedding_tt, None),
    "Mol2Vec": ("mol2vec", Embeddings.gerar_embedding_mol2vec, None),
    "ChemGPT": ("chemgpt", Embeddings.gerar_embedding_chemgpt, None),
    "ChemFM": ("chemfm", Embeddings.gerar_embedding_chemfm, None),
    "MAT": ("mat", Embeddings.gerar_embedding_mat, None),
    "R-MAT": ("rmat", Embeddings.gerar_embedding_rmat, None),
    "CLAMP": ("clamp", Embeddings.gerar_embedding_clamp, None),
}


def caminho(saida, chave, prefixo="Embeddings"):
    return os.path.join(saida, f"{prefixo}_{chave}.npy")


def pronto(saida, chave, n):
    arquivo = caminho(saida, chave)
    return os.path.exists(arquivo) and np.load(arquivo, mmap_mode="r").shape[0] == n


def gerar(nome, smiles):
    """Embeddings de um encoder para a lista de SMILES, uma linha por molécula."""
    _, por_molecula, em_lote = ENCODERS[nome]
    if em_lote is not None:
        return np.asarray(em_lote(smiles), dtype=np.float32), []
    vetores, falhas = [], []
    for i, smile in enumerate(smiles):
        try:
            vetor = por_molecula(smile)
        except Exception as erro:          # uma molécula problemática não derruba o encoder
            vetor, motivo = None, f"{type(erro).__name__}: {erro}"
        else:
            motivo = "o encoder não devolveu vetor"
        if vetor is None:
            falhas.append((i, smile, motivo))
            vetores.append(None)
        else:
            vetores.append(np.asarray(vetor, dtype=np.float32).ravel())
    dimensao = next((len(v) for v in vetores if v is not None), None)
    if dimensao is None:
        raise RuntimeError(f"{nome}: nenhuma molécula foi representada ({falhas[0][2]})")
    # Molécula sem representação vira vetor de zeros, para manter o alinhamento com o CSV;
    # a lista fica em erros_<chave>.txt.
    matriz = np.vstack([v if v is not None else np.zeros(dimensao, np.float32) for v in vetores])
    return matriz, falhas


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--csv", required=True)
    parser.add_argument("--coluna", default="smiles")
    parser.add_argument("--rotulo", default="label")
    parser.add_argument("--saida", required=True)
    parser.add_argument("--encoders", nargs="*", default=list(ENCODERS))
    parser.add_argument("--forcar", action="store_true", help="gera de novo mesmo se já existir")
    args = parser.parse_args()

    base = pd.read_csv(args.csv)
    smiles = base[args.coluna].astype(str).tolist()
    rotulos = base[args.rotulo].to_numpy()
    os.makedirs(args.saida, exist_ok=True)

    status = {}
    for nome in args.encoders:
        chave = ENCODERS[nome][0]
        if not args.forcar and pronto(args.saida, chave, len(smiles)):
            print(f"[checkpoint] {nome}: já existe, pulando", flush=True)
            status[nome] = "existente"
            continue
        inicio = time.time()
        try:
            matriz, falhas = gerar(nome, smiles)
        except Exception as erro:
            print(f"[erro] {nome}: {type(erro).__name__}: {erro}", flush=True)
            status[nome] = f"erro: {type(erro).__name__}"
            continue
        np.save(caminho(args.saida, chave), matriz)
        np.save(caminho(args.saida, chave, "rotulos"), rotulos)
        if falhas:
            with open(os.path.join(args.saida, f"erros_{chave}.txt"), "w") as arquivo:
                for i, smile, motivo in falhas:
                    arquivo.write(f"{i}\t{smile}\t{motivo}\n")
        print(f"[ok] {nome}: {matriz.shape} em {time.time() - inicio:.0f}s"
              + (f", {len(falhas)} moléculas sem vetor" if falhas else ""), flush=True)
        status[nome] = "gerado"

    faltando = [n for n, s in status.items() if s.startswith("erro")]
    print(f"Resumo: {sum(s == 'gerado' for s in status.values())} gerados, "
          f"{sum(s == 'existente' for s in status.values())} já existentes, "
          f"{len(faltando)} com erro {faltando if faltando else ''}")
    sys.exit(1 if faltando else 0)


if __name__ == "__main__":
    main()
