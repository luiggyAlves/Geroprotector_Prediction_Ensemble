"""Gera as figuras do artigo do EX_033 em PDF vetorial, com rótulos em inglês.

As figuras do notebook são PNG a 150 dpi e têm os rótulos em português, que é a língua do
relatório. O artigo é em inglês e vai para composição em LaTeX, então aqui as mesmas figuras
são redesenhadas como PDF vetorial (texto selecionável, sem perda ao ampliar), com fonte serif
para casar com o corpo do texto.

Nada é reajustado: as predições vêm dos checkpoints do EX_032, os mesmos que o notebook usa.

Uso:
    python utils/regressao_logistica/figuras_artigo_ex033.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(AQUI, "..", ".."))
sys.path.insert(0, AQUI)

import ex033_utils as U  # noqa: E402

DIR_BASES = os.path.join(ROOT, "dados", "Bases de dados", "Treinamento - limpas")
BASE_ORIGINAL = os.path.join(DIR_BASES, "Base_GeroLLM_reconstruida", "primary_dataset_dedup.csv")
DIR_PADRONIZADA = os.path.join(DIR_BASES, "Base_GeroLLM_padronizada")
SAIDA = os.path.join(ROOT, "artigo", "figuras")

SEEDS = (42, 43, 44)
N_SPLITS = 5
NOMES = ["MolFormer-XL", "SELFormer", "ChemBERTa-2", "ChemBERTa-3", "MolDeBERTa", "MolCLR",
         "Mole-BERT", "GIN", "GraphMVP", "Uni-Mol", "ECFP4-2048", "Atom-Pair", "Graph-FP",
         "TT", "Mol2Vec", "ChemGPT", "ChemFM", "CLAMP"]
NOME_BASE_EN = {"padronizada": "standardized dataset", "original": "unstandardized dataset"}
MEDIDAS_EN = {"Double-fault": "Double fault", "Q-statistic": "Yule's $Q$",
              "Disagreement": "Disagreement", "Correlation coefficient": "Correlation"}

# Fonte serif e tamanhos pensados para a figura entrar a \textwidth em duas colunas.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 150,
    "pdf.fonttype": 42,        # TrueType: o texto fica selecionável no PDF
    "savefig.bbox": "tight",
})


def carregar():
    """Rótulos, grupos de scaffold, folds e predições de teste dos membros, por base."""
    original = pd.read_csv(BASE_ORIGINAL)
    padronizada = pd.read_csv(os.path.join(DIR_PADRONIZADA, "primary_padronizada.csv"))
    quimica = {
        "padronizada": pd.read_csv(os.path.join(DIR_PADRONIZADA, "padronizacao_base_padronizada.csv")),
        "original": pd.read_csv(os.path.join(DIR_PADRONIZADA, "padronizacao_linhas_originais.csv")),
    }
    Y = {"padronizada": padronizada["label"].astype(int).to_numpy(),
         "original": original["label"].astype(int).to_numpy()}
    grupos = {b: quimica[b]["group"].to_numpy() for b in Y}

    checkpoints = U.Checkpoints(os.path.join(ROOT, "resultados", "checkpoints", "checkpoints_ex032"))
    indices, membros, ensembles = {}, {}, {}
    for base in Y:
        folds = U.repeated_scaffold_folds(Y[base], grupos[base], n_splits=N_SPLITS, seeds=SEEDS)
        indices[base] = {(s, k): (tr, te) for s, k, tr, te in folds}
        membros[base] = {(s, k, n): checkpoints.load(f"membro_{base}_scaffold_seed{s}_fold{k}_{n}")
                         for (s, k) in indices[base] for n in NOMES}
        ensembles[base] = {}
        for (s, k), (tr, te) in indices[base].items():
            registros = {n: membros[base][(s, k, n)] for n in NOMES}
            for nome, registro in U.ensembles_do_fold(Y[base], tr, registros, NOMES, s).items():
                ensembles[base][(s, k, nome)] = registro
        print(f"[ok] {base}: {len(membros[base])} membros, {len(ensembles[base])} ensembles")
    return Y, grupos, indices, membros, ensembles


def figura_diversidade(Y, indices, membros, ensembles, base="padronizada"):
    """Matrizes de diversidade par a par entre os membros de consenso."""
    consenso = U.frequencia_selecao(ensembles[base], NOMES).query("frequencia >= 0.5").index.tolist()
    acertos = U.acertos_oof(Y[base], indices[base], membros[base], consenso, seed=SEEDS[0])

    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.9))
    for ax, (titulo, medida) in zip(axes, U.MEDIDAS_DIVERSIDADE.items()):
        matriz = U.pairwise_matrix(medida, acertos, consenso)
        fora = ~np.eye(len(consenso), dtype=bool)
        imagem = ax.imshow(np.where(fora, matriz, np.nan), cmap="viridis")
        ax.set_xticks(range(len(consenso)), consenso, rotation=45, ha="right")
        ax.set_yticks(range(len(consenso)), consenso)
        meio = np.nanmean(np.where(fora, matriz, np.nan))
        for i in range(len(consenso)):
            for j in range(len(consenso)):
                if i != j:
                    ax.text(j, i, f"{matriz[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if matriz[i, j] < meio else "black")
        ax.set_title(MEDIDAS_EN[titulo])
        fig.colorbar(imagem, ax=ax, fraction=0.046)
    fig.tight_layout()
    salvar(fig, f"ex033_diversity_{base}")


def figura_limiar(Y, indices, membros, ensembles):
    """F1 contra o limiar, curvas OOF e de teste, nas duas bases."""
    melhor = {"padronizada": "MolFormer-XL", "original": "MolFormer-XL"}
    bases = ["padronizada", "original"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.5), sharey=True)
    for ax, base in zip(axes, bases):
        for modelo, rotulo, cor in ((melhor[base], melhor[base], "tab:blue"),
                                    ("Seleção gulosa", "Greedy (F1) ensemble", "tab:red")):
            fonte = ensembles[base] if modelo in U.ENSEMBLES_033 else {
                (s, k, n): {**r, "inner_score": r["inner_oof"]}
                for (s, k, n), r in membros[base].items()}
            probas = {(s, k): r for (s, k, n), r in fonte.items() if n == modelo}
            _, curva, limiar = U.estudo_limiar(Y[base], indices[base], probas)
            ax.plot(curva.index, curva["f1_oof_mean"], ls="-", color=cor, lw=1.4,
                    label=f"{rotulo} — inner OOF")
            ax.plot(curva.index, curva["f1_teste_mean"], ls="--", color=cor, lw=1.4,
                    label=f"{rotulo} — test")
            ax.scatter([0.5, 0.5], curva.loc[0.5, ["f1_oof_mean", "f1_teste_mean"]], s=46,
                       facecolors="none", edgecolors=cor, linewidths=1.5, zorder=3)
            ax.scatter([limiar, limiar], curva.loc[limiar, ["f1_oof_mean", "f1_teste_mean"]],
                       s=110, marker="*", color=cor, edgecolors="black", linewidths=0.5, zorder=4)
        ax.axvline(0.5, color="gray", ls=":", lw=0.9)
        ax.set_xlabel("decision threshold")
        ax.set_title(NOME_BASE_EN[base].capitalize())
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("mean F1 (15 folds)")
    axes[0].scatter([], [], s=46, facecolors="none", edgecolors="gray", linewidths=1.5,
                    label="threshold 0.5")
    axes[0].scatter([], [], s=110, marker="*", color="gray", edgecolors="black", linewidths=0.5,
                    label="selected (inner-OOF peak)")
    # Legenda abaixo dos eixos: dentro deles ela cobriria as curvas do painel da esquerda.
    alcas, rotulos = axes[0].get_legend_handles_labels()
    fig.legend(alcas, rotulos, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=3,
               frameon=False, columnspacing=1.4, handlelength=2.0)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    salvar(fig, "ex033_threshold_scaffold")


def salvar(fig, nome):
    os.makedirs(SAIDA, exist_ok=True)
    caminho = os.path.join(SAIDA, f"{nome}.pdf")
    fig.savefig(caminho)
    fig.savefig(os.path.join(SAIDA, f"{nome}.png"), dpi=300)   # cópia raster para conferência
    plt.close(fig)
    print(f"[ok] {caminho} ({os.path.getsize(caminho) / 1024:.0f} kB)")


if __name__ == "__main__":
    Y, grupos, indices, membros, ensembles = carregar()
    figura_diversidade(Y, indices, membros, ensembles, base="padronizada")
    figura_diversidade(Y, indices, membros, ensembles, base="original")
    figura_limiar(Y, indices, membros, ensembles)
