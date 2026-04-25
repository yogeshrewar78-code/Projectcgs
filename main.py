"""
Main pipeline: Load UD data -> Extract BERT attention -> Evaluate against gold deps -> Visualize.
"""

import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from data_loader import load_conllu
from attention_extractor import load_model, extract_attention
from evaluation import (
    compute_uas_per_head,
    compute_uas_by_deprel,
    aggregate_deprel_results,
    baseline_positional,
    baseline_random,
)
from visualize import (
    plot_uas_heatmap,
    plot_deprel_heatmaps,
    plot_best_heads_bar,
    plot_attention_with_deps,
    plot_cross_lingual_comparison,
    plot_layer_avg_uas,
)


def run_analysis(
    language: str,
    model_name: str,
    max_sentences: int,
    aggregation: str,
):
    """Run full analysis for a single language."""
    print(f"\n{'='*60}")
    print(f"Language: {language} | Model: {model_name} | Sentences: {max_sentences}")
    print(f"{'='*60}")

    # Load data
    print("Loading UD data...")
    sentences = load_conllu(language, max_sentences=max_sentences)
    print(f"Loaded {len(sentences)} sentences")

    # Load model
    print("Loading model...")
    tokenizer, model = load_model(model_name)

    # Extract attention and evaluate
    all_uas = []
    all_deprel_results = defaultdict(list)
    baseline_pos_scores = []
    baseline_rand_scores = []
    rng = np.random.default_rng(42)
    example_sent = None
    example_attention = None

    print("Processing sentences...")
    skipped = 0
    for sent in tqdm(sentences):
        result = extract_attention(tokenizer, model, sent, aggregation=aggregation)
        if result is None:
            skipped += 1
            continue

        attention, _ = result

        # UAS per head
        uas = compute_uas_per_head(attention, sent, directed=True)
        all_uas.append(uas)

        # Per-relation results
        deprel_results = compute_uas_by_deprel(attention, sent)
        for deprel, records in deprel_results.items():
            all_deprel_results[deprel].extend(records)

        # Baselines
        baseline_pos_scores.append(baseline_positional(sent, offset=-1))
        baseline_rand_scores.append(baseline_random(sent, rng))

        # Save first sentence for example visualization
        if example_sent is None:
            example_sent = sent
            example_attention = attention

    if skipped > 0:
        print(f"Skipped {skipped} sentences (too long after tokenization)")

    if not all_uas:
        print("No sentences processed. Exiting.")
        return None

    # Aggregate results
    avg_uas = np.mean(all_uas, axis=0)
    best_layer, best_head = np.unravel_index(avg_uas.argmax(), avg_uas.shape)
    best_uas = avg_uas[best_layer, best_head]
    avg_baseline_pos = np.mean(baseline_pos_scores)
    avg_baseline_rand = np.mean(baseline_rand_scores)

    print(f"\n--- Results ({language}) ---")
    print(f"Best head: Layer {best_layer}, Head {best_head}")
    print(f"Best UAS (directed): {best_uas:.4f}")
    print(f"Positional baseline (prev word): {avg_baseline_pos:.4f}")
    print(f"Random baseline: {avg_baseline_rand:.4f}")

    # Top 5 heads
    flat_indices = np.argsort(avg_uas.ravel())[::-1][:5]
    print("\nTop 5 heads:")
    for rank, flat_idx in enumerate(flat_indices):
        l, h = np.unravel_index(flat_idx, avg_uas.shape)
        print(f"  {rank+1}. Layer {l}, Head {h}: UAS = {avg_uas[l, h]:.4f}")

    # Aggregate deprel results
    num_layers = avg_uas.shape[0]
    num_heads = avg_uas.shape[1]
    deprel_accuracies = aggregate_deprel_results(all_deprel_results, num_layers, num_heads)

    # Print per-relation best heads
    print("\nBest head per relation type:")
    sorted_deprels = sorted(
        deprel_accuracies.keys(),
        key=lambda d: len(all_deprel_results[d]),
        reverse=True,
    )
    for deprel in sorted_deprels[:10]:
        mat = deprel_accuracies[deprel]
        bl, bh = np.unravel_index(mat.argmax(), mat.shape)
        count = len(all_deprel_results[deprel]) // (num_layers * num_heads)
        print(f"  {deprel:20s}: L{bl}H{bh} = {mat.max():.4f}  (n={count})")

    # Generate visualizations
    suffix = f"_{language}"

    plot_uas_heatmap(avg_uas, f"UAS per Layer and Head ({language})", f"uas_heatmap{suffix}.png")
    plot_layer_avg_uas(avg_uas, f"layer_avg_uas{suffix}.png")
    plot_deprel_heatmaps(deprel_accuracies, top_n=6, filename=f"deprel_heatmaps{suffix}.png")
    plot_best_heads_bar(deprel_accuracies, top_n=10, filename=f"best_heads_per_relation{suffix}.png")

    if example_sent is not None and example_attention is not None:
        plot_attention_with_deps(
            example_attention[best_layer, best_head],
            example_sent["words"],
            example_sent["heads"],
            example_sent["deprels"],
            best_layer,
            best_head,
            filename=f"attention_example{suffix}.png",
        )

    return {
        "language": language,
        "best_uas": best_uas,
        "best_layer": best_layer,
        "best_head": best_head,
        "baseline_pos": avg_baseline_pos,
        "baseline_rand": avg_baseline_rand,
        "avg_uas_matrix": avg_uas,
        "deprel_accuracies": deprel_accuracies,
    }


def main():
    parser = argparse.ArgumentParser(description="Dependency Grammar in LLM Attention")
    parser.add_argument(
        "--languages", nargs="+", default=["en"],
        help="Languages to analyze (en, hi, es, fr, de, zh)",
    )
    parser.add_argument(
        "--model", default="bert-base-uncased",
        help="HuggingFace model name (use bert-base-multilingual-cased for non-English)",
    )
    parser.add_argument(
        "--max-sentences", type=int, default=100,
        help="Max sentences per language",
    )
    parser.add_argument(
        "--aggregation", default="mean", choices=["mean", "first", "max"],
        help="Subword attention aggregation method",
    )
    args = parser.parse_args()

    all_results = {}

    for lang in args.languages:
        # Use multilingual BERT for non-English
        model_name = args.model
        if lang != "en" and model_name == "bert-base-uncased":
            model_name = "bert-base-multilingual-cased"
            print(f"Switching to {model_name} for {lang}")

        result = run_analysis(lang, model_name, args.max_sentences, args.aggregation)
        if result:
            all_results[lang] = result

    # Cross-lingual comparison
    if len(all_results) > 1:
        lang_scores = {lang: r["best_uas"] for lang, r in all_results.items()}
        baseline_scores = {lang: r["baseline_pos"] for lang, r in all_results.items()}
        plot_cross_lingual_comparison(lang_scores, baseline_scores)

        print(f"\n{'='*60}")
        print("Cross-lingual Summary")
        print(f"{'='*60}")
        for lang, r in all_results.items():
            print(f"  {lang}: Best UAS = {r['best_uas']:.4f} (L{r['best_layer']}H{r['best_head']})")

    print("\nDone! Check the 'outputs/' directory for visualizations.")


if __name__ == "__main__":
    main()
