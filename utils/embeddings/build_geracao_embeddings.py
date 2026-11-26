"""Gera o notebook de geração de embeddings.

Escreve notebooks/geracao_embeddings.ipynb. Substitui os quatro notebooks antigos de
notebooks/embeddings/, que divergiam entre si em caminhos, encoders e tratamento de falhas.

Uso:
    python utils/embeddings/build_geracao_embeddings.py
"""

import json
import os

NOTEBOOK = "geracao_embeddings.ipynb"
cells = []


def md(texto):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n")})


def code(texto):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": texto.strip("\n")})


# =========================================================================================
md("""
# Geração dos embeddings dos 18 encoders

Este notebook é o painel de controle dos embeddings do estudo: ele diz o que já está em disco,
gera o que falta e confere o que foi gerado. Uma linha por molécula, na ordem do CSV — é esse
alinhamento que os notebooks de treino (EX_032, EX_033) e a triagem da base Eterna + NuBBE
assumem ao carregar os `.npy`.

O cálculo em si **não** vive aqui. Ele está em `utils/embeddings/`:

- `Embeddings.py` — uma função por encoder, com os checkpoints e as featurizações;
- `gerar_embeddings_base.py` — a CLI que percorre uma base, chama o encoder certo e grava
  `Embeddings_<chave>.npy` + `rotulos_<chave>.npy` (e `erros_<chave>.txt`, quando alguma
  molécula não é representada).

Este notebook só orquestra essa CLI. Isso é de propósito: os encoders não cabem no env do
kernel (torch, transformers, torch_geometric, gensim, selfies, unimol_tools), então rodam num
env conda à parte, em subprocesso. Era aí que os notebooks antigos desta pasta divergiam entre
si — cada um importava `Embeddings` direto, com caminhos do Windows fixos no código e uma lista
de encoders diferente. Os quatro foram substituídos por este.

**Checkpoint por encoder:** se o `.npy` já existe com o número certo de linhas, ele é pulado.
Rodar o notebook inteiro de novo não recalcula nada, e uma execução interrompida continua de
onde parou.
""")

# ---------------------------------------------------------------------------- 1
md("""
## 1. Configuração

Os 18 encoders são os do EX_032/EX_033 — a mesma lista, com as mesmas chaves de arquivo, para
que os `.npy` sejam lidos lá sem tradução de nomes.

MAT e R-MAT, que existiam no EX_027, ficam de fora: dependem do pacote `huggingmolecules`, que
só existe no GitHub e não entra no env dos encoders. `gerar_embeddings_base.py` ainda os
conhece, então basta instalar o pacote e acrescentá-los ao dicionário abaixo para voltarem.

As duas bases são as **padronizadas** (`rdMolStandardize`, a química do EX_033). As pastas
`Embeddings_GeroLLM` e `Embeddings_GeroLLM_testeExterno`, sem sufixo, são as da base não
padronizada do EX_027: ficam como estão, e este notebook não escreve nelas.

A base de triagem (Eterna + NuBBE) tem o fluxo dela em
`teste_eterna/notebooks/`, que chama a mesma CLI — não é gerada aqui.
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

# Este notebook roda de notebooks/.
ROOT = os.path.abspath("..")                          # desenvolvimento_modelos/
DIR_BASES = os.path.join(ROOT, "dados", "Bases de dados", "Treinamento - limpas",
                         "Base_GeroLLM_padronizada")
DIR_SALVOS = os.path.join(ROOT, "dados", "Embeddings_salvos")
SCRIPT_EMBEDDINGS = os.path.join(ROOT, "utils", "embeddings", "gerar_embeddings_base.py")

ENCODERS = {  # nome no estudo -> chave do arquivo de embedding (igual ao EX_032/EX_033)
    "MolFormer-XL": "molformer", "SELFormer": "selformer", "ChemBERTa-2": "chembert",
    "ChemBERTa-3": "chemberta3", "MolDeBERTa": "moldeberta", "MolCLR": "molclr",
    "Mole-BERT": "molebert", "GIN": "gin", "GraphMVP": "graphmvp", "Uni-Mol": "unimol",
    "ECFP4-2048": "ecfp4", "Atom-Pair": "atompair", "Graph-FP": "graphfp", "TT": "tt",
    "Mol2Vec": "mol2vec", "ChemGPT": "chemgpt", "ChemFM": "chemfm", "CLAMP": "clamp",
}
BINARIOS = {"ECFP4-2048", "Atom-Pair", "Graph-FP", "TT"}   # fingerprints, não embeddings densos

# Uma entrada por base: de qual CSV ela vem e em qual pasta os .npy ficam.
BASES = {
    "treino": {"csv": os.path.join(DIR_BASES, "primary_padronizada.csv"),
               "saida": os.path.join(DIR_SALVOS, "Embeddings_GeroLLM_padronizada")},
    "externo": {"csv": os.path.join(DIR_BASES, "externo_padronizado.csv"),
                "saida": os.path.join(DIR_SALVOS, "Embeddings_GeroLLM_testeExterno_padronizada")},
}
COLUNA_SMILES = "smiles"
COLUNA_ROTULO = "label"

for nome, info in BASES.items():
    base = pd.read_csv(info["csv"])
    info["n"] = len(base)
    info["rotulos"] = base[COLUNA_ROTULO].to_numpy()
    os.makedirs(info["saida"], exist_ok=True)
    print(f"[{nome}] {info['n']} moléculas, {info['rotulos'].mean():.1%} positivas "
          f"— {os.path.basename(info['csv'])}")
print(f"\\n{len(ENCODERS)} encoders; CLI: {os.path.relpath(SCRIPT_EMBEDDINGS, ROOT)}")
""")

# ---------------------------------------------------------------------------- 2
md("""
## 2. O que já está em disco

Um encoder conta como pronto quando o `.npy` existe **e** tem uma linha por molécula do CSV. O
número de linhas é a conferência que importa: um arquivo gerado para uma versão anterior da
base sobrevive em disco com o nome certo e o tamanho errado, e seria carregado desalinhado dos
rótulos.
""")
code("""
def arquivos(base, chave):
    pasta = BASES[base]["saida"]
    return (os.path.join(pasta, f"Embeddings_{chave}.npy"),
            os.path.join(pasta, f"rotulos_{chave}.npy"))


def pronto(base, nome):
    arquivo = arquivos(base, ENCODERS[nome])[0]
    return (os.path.exists(arquivo)
            and np.load(arquivo, mmap_mode="r").shape[0] == BASES[base]["n"])


def estado(base, nome):
    arquivo = arquivos(base, ENCODERS[nome])[0]
    if not os.path.exists(arquivo):
        return "ausente"
    n = np.load(arquivo, mmap_mode="r").shape[0]
    if n != BASES[base]["n"]:
        return f"desalinhado ({n} linhas, base tem {BASES[base]['n']})"
    return "ok"


inventario = pd.DataFrame({base: {nome: estado(base, nome) for nome in ENCODERS}
                           for base in BASES})
display(inventario)

FALTANDO = {base: [n for n in ENCODERS if not pronto(base, n)] for base in BASES}
for base, lista in FALTANDO.items():
    print(f"[{base}] {len(ENCODERS) - len(lista)}/{len(ENCODERS)} prontos"
          + (f"; falta gerar: {', '.join(lista)}" if lista else ""))
""")

# ---------------------------------------------------------------------------- 3
md("""
## 3. Env conda dos encoders

Os encoders pedem torch, transformers, torch_geometric, gensim, selfies e unimol_tools, que não
cabem no env do kernel. Eles rodam num env conda à parte, `embeddings`, descoberto pelo `conda`
do PATH — e criado na hora se não existir, com os pacotes instalados.

Descobrir em vez de fixar o caminho é o que faz este notebook rodar na máquina em que o env está
em `~/miniconda3` e na que ele está sob o Homebrew. A célula só faz trabalho se houver embedding
para gerar.
""")
code("""
ENV_EMBEDDINGS = "embeddings"
REQUISITOS_EMBEDDINGS = [
    "numpy>=2", "pandas", "scikit-learn", "matplotlib", "rdkit", "torch", "transformers",
    "tokenizers", "safetensors", "huggingface-hub", "torch_geometric", "gensim", "selfies",
    "unimol-tools",
]


def python_do_env_embeddings():
    conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
    assert conda, "conda não encontrado no PATH; ele é necessário para o env dos encoders"
    def envs():
        saida = subprocess.run([conda, "env", "list", "--json"], capture_output=True,
                               text=True, check=True).stdout
        return json.loads(saida)["envs"]
    prefixo = next((e for e in envs() if os.path.basename(e) == ENV_EMBEDDINGS), None)
    if prefixo is None:
        print(f"[env] criando o env conda '{ENV_EMBEDDINGS}' (python 3.11)")
        subprocess.run([conda, "create", "-y", "-q", "-n", ENV_EMBEDDINGS, "python=3.11"],
                       check=True)
        prefixo = next(e for e in envs() if os.path.basename(e) == ENV_EMBEDDINGS)
    python = os.path.join(prefixo, "bin", "python")
    print(f"[env] instalando/conferindo os pacotes dos encoders em {python}")
    subprocess.run([python, "-m", "pip", "install", "-q", *REQUISITOS_EMBEDDINGS], check=True)
    return python


PYTHON_EMBEDDINGS = (python_do_env_embeddings()
                     if any(FALTANDO.values()) else None)
print("Nada a gerar; o env não foi tocado." if PYTHON_EMBEDDINGS is None else "[env] pronto")
""")

# ---------------------------------------------------------------------------- 4
md("""
## 4. Geração

Uma chamada da CLI por base, só com os encoders que faltam. A saída vem filtrada nas linhas que
importam (`[ok]`, `[erro]`, `[checkpoint]`, `Resumo`), porque o progresso dos encoders grandes
enche a saída de barras de download.

Para refazer um encoder de propósito — checkpoint novo, base corrigida — passe o nome em
`REGERAR`: ele entra na chamada mesmo estando pronto, com `--forcar`.
""")
code("""
REGERAR = []         # ex.: ["Uni-Mol", "CLAMP"]; [] = gerar só o que falta

for base, lista in FALTANDO.items():
    gerar = lista + [n for n in REGERAR if n not in lista]
    if not gerar:
        print(f"[{base}] nada a fazer")
        continue
    info = BASES[base]
    print(f"\\n[{base}] gerando: {', '.join(gerar)}")
    comando = [PYTHON_EMBEDDINGS, SCRIPT_EMBEDDINGS, "--csv", info["csv"],
               "--coluna", COLUNA_SMILES, "--rotulo", COLUNA_ROTULO,
               "--saida", info["saida"], "--encoders", *gerar]
    if REGERAR:
        comando.append("--forcar")
    processo = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True)
    for linha in processo.stdout:
        if linha.startswith(("[ok]", "[erro]", "[checkpoint]", "Resumo")):
            print(linha, end="")
    processo.wait()
    print(f"[{base}] código de saída {processo.returncode}")
""")

# ---------------------------------------------------------------------------- 5
md("""
## 5. Conferência

Três coisas podem estar erradas num `.npy` que existe e tem o tamanho certo, e as três
quebrariam o treino de um jeito silencioso:

1. **rótulos trocados** — `rotulos_<chave>.npy` tem que ser idêntico à coluna `label` do CSV,
   na mesma ordem. É isso que autoriza usar um vetor `y` só para todos os `X`;
2. **moléculas sem vetor** — a CLI grava zeros na linha da molécula que o encoder não
   representou, para não desalinhar o resto, e lista o caso em `erros_<chave>.txt`. Uma linha
   de zeros é um composto sem informação, não um erro de leitura;
3. **dimensão inesperada** — serve para notar um checkpoint diferente do usado antes.
""")
code("""
conferencia = []
for base, info in BASES.items():
    for nome, chave in ENCODERS.items():
        arquivo_x, arquivo_y = arquivos(base, chave)
        if not os.path.exists(arquivo_x):
            conferencia.append({"base": base, "encoder": nome, "linhas": "—"})
            continue
        X = np.load(arquivo_x)
        y = np.load(arquivo_y) if os.path.exists(arquivo_y) else None
        erros = os.path.join(info["saida"], f"erros_{chave}.txt")
        conferencia.append({
            "base": base, "encoder": nome, "linhas": X.shape[0], "dim": X.shape[1],
            "linhas == base": X.shape[0] == info["n"],
            "rótulos == CSV": (y is not None and y.shape == info["rotulos"].shape
                               and bool((y == info["rotulos"]).all())),
            "linhas zeradas": int((~X.any(axis=1)).sum()),
            "em erros_*.txt": sum(1 for _ in open(erros)) if os.path.exists(erros) else 0,
        })

conferencia = pd.DataFrame(conferencia).set_index(["base", "encoder"])
display(conferencia)

problemas = conferencia[(conferencia["linhas == base"] != True)
                        | (conferencia["rótulos == CSV"] != True)]
print(f"{len(conferencia) - len(problemas)} de {len(conferencia)} pares (base, encoder) "
      f"com embedding alinhado ao CSV")
if len(problemas):
    print("Sem embedding ou desalinhado — não carregar no treino antes de gerar:")
    display(problemas)
""")

md("""
## Notas

- **O contrato de alinhamento.** Uma linha por linha do CSV, na ordem do CSV, inclusive para a
  molécula que falhou (linha de zeros). Nada é filtrado na geração. Os notebooks antigos desta
  pasta descartavam as moléculas problemáticas de `X` e de `y` juntas, o que dava um `y`
  diferente por encoder — e era a origem do aviso, na célula final deles, de não usar um vetor
  de rótulos só para todos os `X`. Com o contrato atual, usar um `y` só é o certo, e a seção 5
  é quem verifica isso.
- **Os quatro fingerprints** (ECFP4-2048, Atom-Pair, Graph-FP, TT) são vetores binários do
  RDKit, não embeddings densos: rodam em segundos e não precisam de GPU. Os outros 14 baixam
  checkpoint do HF Hub na primeira vez.
- **ChemGPT** usa a variante de 4,7M parâmetros (128 dimensões), fixada na CLI — a mesma do
  EX_027, para as comparações entre experimentos continuarem válidas.
- **Uni-Mol** é o único que roda em lote, e precisa de conformação 3D; quando não consegue
  gerar, cai num fallback 2D. No EX_033 isso afetou 1,1% dos compostos.
- **Gerar de novo custa caro** e não muda resultado: os encoders são congelados, sem fine-tune.
  Só vale refazer quando a base muda (foi o caso da padronização) ou quando o checkpoint de um
  encoder muda.
""")

# =========================================================================================
for posicao, cell in enumerate(cells):
    cell["id"] = f"embeddings-{posicao:03d}"
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
