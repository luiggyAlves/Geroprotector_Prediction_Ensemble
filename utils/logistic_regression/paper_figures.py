"""Redraws the paper figures as vector PDFs.

The notebook saves its figures as 150 dpi PNGs, sized for on-screen reading. The paper is
typeset in LaTeX, so the same figures are redrawn here as vector PDFs (selectable text, no loss
when zoomed), with a serif face to match the body text.

Nothing is refitted: the predictions come from the stored member checkpoints, the same ones the
notebook uses.

Usage:
    python utils/logistic_regression/paper_figures.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import analysis_utils as U  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
UNSTANDARDIZED_CSV = os.path.join(DATA_DIR, "gerollm_reconstructed", "primary_dataset_dedup.csv")
STANDARDIZED_DIR = os.path.join(DATA_DIR, "gerollm_standardized")
OUT_DIR = os.path.join(ROOT, "results", "figures", "paper")

SEEDS = (42, 43, 44)
N_SPLITS = 5
NAMES = ["MolFormer-XL", "SELFormer", "ChemBERTa-2", "ChemBERTa-3", "MolDeBERTa", "MolCLR",
         "Mole-BERT", "GIN", "GraphMVP", "Uni-Mol", "ECFP4-2048", "Atom-Pair", "Graph-FP",
         "TT", "Mol2Vec", "ChemGPT", "ChemFM", "CLAMP"]
DATASET_LABEL = {"standardized": "standardized dataset",
                 "unstandardized": "unstandardized dataset"}
MEASURE_LABEL = {"Double-fault": "Double fault", "Q-statistic": "Yule's $Q$",
                 "Disagreement": "Disagreement", "Correlation coefficient": "Correlation"}

# Serif face and sizes chosen so that each figure fits \textwidth in a two-column layout.
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
    "pdf.fonttype": 42,        # TrueType, so the text stays selectable in the PDF
    "savefig.bbox": "tight",
})


def load():
    """Labels, scaffold groups, folds and member test predictions, per dataset."""
    unstandardized = pd.read_csv(UNSTANDARDIZED_CSV)
    standardized = pd.read_csv(os.path.join(STANDARDIZED_DIR, "primary_standardized.csv"))
    chemistry = {
        "standardized": pd.read_csv(os.path.join(
            STANDARDIZED_DIR, "standardization_audit_standardized.csv")),
        "unstandardized": pd.read_csv(os.path.join(
            STANDARDIZED_DIR, "standardization_audit_unstandardized_rows.csv")),
    }
    Y = {"standardized": standardized["label"].astype(int).to_numpy(),
         "unstandardized": unstandardized["label"].astype(int).to_numpy()}
    groups = {d: chemistry[d]["group"].to_numpy() for d in Y}

    checkpoints = U.Checkpoints(os.path.join(ROOT, "results", "checkpoints", "member_fits"))
    indices, members, ensembles = {}, {}, {}
    for dataset in Y:
        folds = U.repeated_scaffold_folds(Y[dataset], groups[dataset], n_splits=N_SPLITS,
                                          seeds=SEEDS)
        indices[dataset] = {(s, k): (tr, te) for s, k, tr, te in folds}
        members[dataset] = {
            (s, k, n): checkpoints.load(f"member_{dataset}_scaffold_seed{s}_fold{k}_{n}")
            for (s, k) in indices[dataset] for n in NAMES}
        ensembles[dataset] = {}
        for (s, k), (tr, te) in indices[dataset].items():
            records = {n: members[dataset][(s, k, n)] for n in NAMES}
            for name, record in U.fold_ensembles(Y[dataset], tr, records, NAMES, s).items():
                ensembles[dataset][(s, k, name)] = record
        print(f"[ok] {dataset}: {len(members[dataset])} members, "
              f"{len(ensembles[dataset])} ensembles")
    return Y, groups, indices, members, ensembles


def diversity_figure(Y, indices, members, ensembles, dataset="standardized"):
    """Pairwise diversity matrices between the consensus members."""
    consensus = (U.selection_frequency(ensembles[dataset], NAMES)
                 .query("frequency >= 0.5").index.tolist())
    hits = U.oof_hits(Y[dataset], indices[dataset], members[dataset], consensus, seed=SEEDS[0])

    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.9))
    for ax, (title, measure) in zip(axes, U.DIVERSITY_MEASURES.items()):
        matrix = U.pairwise_matrix(measure, hits, consensus)
        off_diagonal = ~np.eye(len(consensus), dtype=bool)
        image = ax.imshow(np.where(off_diagonal, matrix, np.nan), cmap="viridis")
        ax.set_xticks(range(len(consensus)), consensus, rotation=45, ha="right")
        ax.set_yticks(range(len(consensus)), consensus)
        middle = np.nanmean(np.where(off_diagonal, matrix, np.nan))
        for i in range(len(consensus)):
            for j in range(len(consensus)):
                if i != j:
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if matrix[i, j] < middle else "black")
        ax.set_title(MEASURE_LABEL[title])
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    save(fig, f"diversity_{dataset}")


def threshold_figure(Y, indices, members, ensembles):
    """F1 against the decision threshold, out-of-fold and test curves, in both datasets."""
    best = {"standardized": "MolFormer-XL", "unstandardized": "MolFormer-XL"}
    datasets = ["standardized", "unstandardized"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.5), sharey=True)
    for ax, dataset in zip(axes, datasets):
        for model, label, color in ((best[dataset], best[dataset], "tab:blue"),
                                    ("Greedy (F1)", "Greedy (F1) ensemble", "tab:red")):
            source = ensembles[dataset] if model in U.ALL_ENSEMBLES else {
                (s, k, n): {**r, "inner_score": r["inner_oof"]}
                for (s, k, n), r in members[dataset].items()}
            probas = {(s, k): r for (s, k, n), r in source.items() if n == model}
            _, curve, threshold = U.threshold_sweep(Y[dataset], indices[dataset], probas)
            ax.plot(curve.index, curve["f1_oof_mean"], ls="-", color=color, lw=1.4,
                    label=f"{label} — inner out-of-fold")
            ax.plot(curve.index, curve["f1_test_mean"], ls="--", color=color, lw=1.4,
                    label=f"{label} — test")
            ax.scatter([0.5, 0.5], curve.loc[0.5, ["f1_oof_mean", "f1_test_mean"]], s=46,
                       facecolors="none", edgecolors=color, linewidths=1.5, zorder=3)
            ax.scatter([threshold, threshold],
                       curve.loc[threshold, ["f1_oof_mean", "f1_test_mean"]],
                       s=110, marker="*", color=color, edgecolors="black", linewidths=0.5,
                       zorder=4)
        ax.axvline(0.5, color="gray", ls=":", lw=0.9)
        ax.set_xlabel("decision threshold")
        ax.set_title(DATASET_LABEL[dataset].capitalize())
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("mean F1 (15 folds)")
    axes[0].scatter([], [], s=46, facecolors="none", edgecolors="gray", linewidths=1.5,
                    label="threshold 0.5")
    axes[0].scatter([], [], s=110, marker="*", color="gray", edgecolors="black", linewidths=0.5,
                    label="selected (inner out-of-fold peak)")
    # Legend below the axes: inside them it would cover the curves of the left panel.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=3,
               frameon=False, columnspacing=1.4, handlelength=2.0)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "threshold_scaffold")


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.pdf")
    fig.savefig(path)
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=300)   # raster copy for checking
    plt.close(fig)
    print(f"[ok] {path} ({os.path.getsize(path) / 1024:.0f} kB)")


if __name__ == "__main__":
    Y, groups, indices, members, ensembles = load()
    diversity_figure(Y, indices, members, ensembles, dataset="standardized")
    diversity_figure(Y, indices, members, ensembles, dataset="unstandardized")
    threshold_figure(Y, indices, members, ensembles)
