"""
Visualization utilities for dependency grammar analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_uas_heatmap(
    uas_matrix: np.ndarray,
    title: str = "UAS per Layer and Head",
    filename: str = "uas_heatmap.png",
):
    """Plot a layer x head heatmap of UAS scores."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        uas_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=[f"H{i}" for i in range(uas_matrix.shape[1])],
        yticklabels=[f"L{i}" for i in range(uas_matrix.shape[0])],
        ax=ax,
        vmin=0,
        vmax=max(0.5, uas_matrix.max()),
    )
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_deprel_heatmaps(
    deprel_accuracies: Dict[str, np.ndarray],
    top_n: int = 6,
    filename: str = "deprel_heatmaps.png",
):
    """Plot heatmaps for top-N most frequent dependency relation types."""
    # Sort by number of non-zero entries (proxy for frequency)
    sorted_deprels = sorted(
        deprel_accuracies.keys(),
        key=lambda d: deprel_accuracies[d].sum(),
        reverse=True,
    )[:top_n]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, deprel in enumerate(sorted_deprels):
        if idx >= len(axes):
            break
        mat = deprel_accuracies[deprel]
        sns.heatmap(
            mat,
            cmap="YlOrRd",
            xticklabels=[f"H{i}" for i in range(mat.shape[1])],
            yticklabels=[f"L{i}" for i in range(mat.shape[0])],
            ax=axes[idx],
            vmin=0,
            vmax=max(0.5, mat.max()),
        )
        axes[idx].set_title(f"Relation: {deprel}")
        axes[idx].set_xlabel("Head")
        axes[idx].set_ylabel("Layer")

    # Hide unused axes
    for idx in range(len(sorted_deprels), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Per-Relation UAS by Layer and Head", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_best_heads_bar(
    deprel_accuracies: Dict[str, np.ndarray],
    top_n: int = 10,
    filename: str = "best_heads_per_relation.png",
):
    """Bar chart showing the best-performing head for each relation type."""
    sorted_deprels = sorted(
        deprel_accuracies.keys(),
        key=lambda d: deprel_accuracies[d].max(),
        reverse=True,
    )[:top_n]

    deprels = []
    best_scores = []
    labels = []

    for deprel in sorted_deprels:
        mat = deprel_accuracies[deprel]
        best_layer, best_head = np.unravel_index(mat.argmax(), mat.shape)
        deprels.append(deprel)
        best_scores.append(mat.max())
        labels.append(f"L{best_layer}H{best_head}")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(deprels, best_scores, color=sns.color_palette("viridis", len(deprels)))

    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            label,
            ha="center",
            fontsize=8,
        )

    ax.set_xlabel("Dependency Relation")
    ax.set_ylabel("Best Head UAS")
    ax.set_title("Best Attention Head per Dependency Relation")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_attention_with_deps(
    attention_matrix: np.ndarray,
    words: List[str],
    gold_heads: List[int],
    deprels: List[str],
    layer: int,
    head: int,
    filename: str = "attention_example.png",
):
    """Plot attention heatmap for a single sentence with gold deps marked."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        attention_matrix,
        xticklabels=words,
        yticklabels=words,
        cmap="Blues",
        ax=ax,
    )

    # Mark gold dependencies with red dots
    for i, (head_idx, deprel) in enumerate(zip(gold_heads, deprels)):
        if head_idx == 0:
            continue
        j = head_idx - 1
        ax.plot(j + 0.5, i + 0.5, "ro", markersize=8, markeredgecolor="red")

    ax.set_title(f"Attention (Layer {layer}, Head {head}) with gold deps (red dots)")
    ax.set_xlabel("Attended to")
    ax.set_ylabel("Attending from")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_cross_lingual_comparison(
    language_scores: Dict[str, float],
    baseline_scores: Optional[Dict[str, float]] = None,
    filename: str = "cross_lingual.png",
):
    """Bar chart comparing best-head UAS across languages."""
    languages = list(language_scores.keys())
    scores = [language_scores[lang] for lang in languages]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(languages))
    width = 0.35

    ax.bar(x - width / 2, scores, width, label="Best Head UAS", color="steelblue")

    if baseline_scores:
        bl_scores = [baseline_scores.get(lang, 0) for lang in languages]
        ax.bar(x + width / 2, bl_scores, width, label="Positional Baseline", color="lightcoral")

    ax.set_xlabel("Language")
    ax.set_ylabel("UAS")
    ax.set_title("Best Attention Head UAS Across Languages")
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_layer_avg_uas(
    uas_matrix: np.ndarray,
    filename: str = "layer_avg_uas.png",
):
    """Line plot showing average UAS per layer (averaged over heads)."""
    layer_avg = uas_matrix.mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(layer_avg)), layer_avg, "o-", color="steelblue", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Average UAS")
    ax.set_title("Average UAS by Layer")
    ax.set_xticks(range(len(layer_avg)))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")
