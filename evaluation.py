"""
Evaluation metrics for comparing attention patterns with gold dependency structures.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def compute_uas_per_head(
    attention: np.ndarray,
    sentence: Dict,
    directed: bool = False
) -> np.ndarray:
    """
    Compute Undirected Attachment Score (UAS) for each (layer, head).

    For each word, we take the argmax of its attention row (excluding self-attention)
    and check if it matches the gold dependency head.

    Args:
        attention: (num_layers, num_heads, num_words, num_words)
        sentence: dict with 'heads' (1-indexed, 0=root)
        directed: if True, check directed match; if False, check undirected

    Returns:
        (num_layers, num_heads) array of UAS scores for this sentence
    """
    num_layers, num_heads, num_words, _ = attention.shape
    gold_heads = sentence["heads"]
    scores = np.zeros((num_layers, num_heads))

    for layer in range(num_layers):
        for head in range(num_heads):
            attn = attention[layer, head].copy()

            correct = 0
            total = 0

            for word_idx in range(num_words):
                gold_head = gold_heads[word_idx]
                if gold_head == 0:
                    # Root word — skip (no head to predict)
                    continue

                gold_head_idx = gold_head - 1  # convert to 0-indexed

                # Zero out self-attention for argmax
                attn_row = attn[word_idx].copy()
                attn_row[word_idx] = 0

                predicted_head = np.argmax(attn_row)

                if directed:
                    if predicted_head == gold_head_idx:
                        correct += 1
                else:
                    # Undirected: check both directions
                    if predicted_head == gold_head_idx:
                        correct += 1
                    else:
                        # Also check if gold_head attends most to this word
                        attn_row_rev = attn[gold_head_idx].copy()
                        attn_row_rev[gold_head_idx] = 0
                        if np.argmax(attn_row_rev) == word_idx:
                            correct += 1

                total += 1

            if total > 0:
                scores[layer, head] = correct / total

    return scores


def compute_uas_by_deprel(
    attention: np.ndarray,
    sentence: Dict,
) -> Dict[str, List[Tuple[int, int, bool]]]:
    """
    For each dependency relation type, record per-head match results.

    Returns dict: deprel -> list of (layer, head, is_correct) for each word.
    We use directed UAS here (argmax of dependent's attention row = gold head).
    """
    num_layers, num_heads, num_words, _ = attention.shape
    gold_heads = sentence["heads"]
    deprels = sentence["deprels"]

    results = defaultdict(list)

    for word_idx in range(num_words):
        gold_head = gold_heads[word_idx]
        deprel = deprels[word_idx]

        if gold_head == 0:
            continue

        gold_head_idx = gold_head - 1

        for layer in range(num_layers):
            for head in range(num_heads):
                attn_row = attention[layer, head, word_idx].copy()
                attn_row[word_idx] = 0
                predicted_head = np.argmax(attn_row)
                is_correct = (predicted_head == gold_head_idx)
                results[deprel].append((layer, head, is_correct))

    return results


def baseline_positional(sentence: Dict, offset: int = -1) -> float:
    """
    Positional baseline: predict head as the word at position (word_idx + offset).

    Default offset=-1 means "previous word is the head."
    Returns UAS score.
    """
    gold_heads = sentence["heads"]
    num_words = len(gold_heads)

    correct = 0
    total = 0

    for word_idx in range(num_words):
        gold_head = gold_heads[word_idx]
        if gold_head == 0:
            continue

        gold_head_idx = gold_head - 1
        predicted = word_idx + offset
        if 0 <= predicted < num_words:
            if predicted == gold_head_idx:
                correct += 1

        total += 1

    return correct / total if total > 0 else 0.0


def baseline_random(sentence: Dict, rng: np.random.Generator = None) -> float:
    """Random baseline: predict a random word (excluding self) as head."""
    if rng is None:
        rng = np.random.default_rng(42)

    gold_heads = sentence["heads"]
    num_words = len(gold_heads)

    correct = 0
    total = 0

    for word_idx in range(num_words):
        gold_head = gold_heads[word_idx]
        if gold_head == 0:
            continue

        gold_head_idx = gold_head - 1
        candidates = [i for i in range(num_words) if i != word_idx]
        if candidates:
            predicted = rng.choice(candidates)
            if predicted == gold_head_idx:
                correct += 1

        total += 1

    return correct / total if total > 0 else 0.0


def aggregate_deprel_results(
    all_results: Dict[str, List[Tuple[int, int, bool]]],
    num_layers: int = 12,
    num_heads: int = 12,
) -> Dict[str, np.ndarray]:
    """
    Aggregate per-sentence deprel results into per-head accuracy matrices.

    Returns dict: deprel -> (num_layers, num_heads) accuracy array
    """
    aggregated = {}

    for deprel, records in all_results.items():
        scores = np.zeros((num_layers, num_heads))
        counts = np.zeros((num_layers, num_heads))

        for layer, head, is_correct in records:
            counts[layer, head] += 1
            if is_correct:
                scores[layer, head] += 1

        # Avoid division by zero
        mask = counts > 0
        accuracy = np.zeros_like(scores)
        accuracy[mask] = scores[mask] / counts[mask]
        aggregated[deprel] = accuracy

    return aggregated


if __name__ == "__main__":
    from data_loader import load_conllu
    from attention_extractor import load_model, extract_attention

    tokenizer, model = load_model("bert-base-uncased")
    sents = load_conllu("en", max_sentences=5)

    all_uas = []
    baseline_pos_scores = []
    baseline_rand_scores = []
    rng = np.random.default_rng(42)

    for sent in sents:
        result = extract_attention(tokenizer, model, sent)
        if result is None:
            continue
        attention, _ = result

        uas = compute_uas_per_head(attention, sent, directed=True)
        all_uas.append(uas)

        baseline_pos_scores.append(baseline_positional(sent, offset=-1))
        baseline_rand_scores.append(baseline_random(sent, rng))

    avg_uas = np.mean(all_uas, axis=0)
    best_layer, best_head = np.unravel_index(avg_uas.argmax(), avg_uas.shape)

    print(f"Best head: Layer {best_layer}, Head {best_head}, UAS = {avg_uas[best_layer, best_head]:.3f}")
    print(f"Positional baseline (prev word): {np.mean(baseline_pos_scores):.3f}")
    print(f"Random baseline: {np.mean(baseline_rand_scores):.3f}")
