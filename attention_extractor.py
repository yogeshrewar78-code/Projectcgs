"""
Extract attention weights from BERT and align subword tokens to UD words.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional, Tuple


def load_model(model_name: str = "bert-base-uncased"):
    """Load a pretrained transformer model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()
    return tokenizer, model


def align_subwords_to_words(
    tokenizer, words: List[str], tokenizer_output
) -> List[List[int]]:
    """
    Map each UD word to its corresponding subword token indices.

    Returns a list of lists: word_to_subword[i] = [subword indices for word i].
    Indices are into the full token list (including [CLS] and [SEP]).
    """
    # Get word_ids from tokenizer (maps each subtoken to its word index)
    word_ids = tokenizer_output.word_ids()

    # Build mapping: word_index -> list of subword positions
    word_to_subword = {}
    for subword_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue  # skip [CLS], [SEP], padding
        if word_id not in word_to_subword:
            word_to_subword[word_id] = []
        word_to_subword[word_id].append(subword_idx)

    # The tokenizer may split the sentence differently than UD words.
    # We use is_split_into_words=True in the caller to ensure 1:1 mapping.
    result = []
    for i in range(len(words)):
        if i in word_to_subword:
            result.append(word_to_subword[i])
        else:
            result.append([])

    return result


def aggregate_attention(
    attention_matrix: np.ndarray,
    word_to_subword: List[List[int]],
    method: str = "mean"
) -> np.ndarray:
    """
    Aggregate subword-level attention matrix to word-level.

    Args:
        attention_matrix: (num_subtokens, num_subtokens) attention weights
        word_to_subword: mapping from word index to subword indices
        method: 'mean', 'first', or 'max'

    Returns:
        (num_words, num_words) aggregated attention matrix
    """
    num_words = len(word_to_subword)
    word_attn = np.zeros((num_words, num_words))

    for i, src_indices in enumerate(word_to_subword):
        for j, tgt_indices in enumerate(word_to_subword):
            if not src_indices or not tgt_indices:
                continue

            # Extract the subword attention block
            block = attention_matrix[np.ix_(src_indices, tgt_indices)]

            if method == "mean":
                word_attn[i, j] = block.mean()
            elif method == "first":
                word_attn[i, j] = block[0, 0]
            elif method == "max":
                word_attn[i, j] = block.max()

    return word_attn


def extract_attention(
    tokenizer,
    model,
    sentence: Dict,
    aggregation: str = "mean"
) -> Optional[Tuple[np.ndarray, List[List[int]]]]:
    """
    Extract word-level attention matrices for a sentence.

    Args:
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        sentence: dict with 'words' key (list of word strings)
        aggregation: how to aggregate subword attention ('mean', 'first', 'max')

    Returns:
        Tuple of:
            - attention: np.ndarray of shape (num_layers, num_heads, num_words, num_words)
            - word_to_subword: the alignment mapping
        Returns None if sentence is too long for the model.
    """
    words = sentence["words"]

    # Tokenize with is_split_into_words=True to preserve word boundaries
    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    # Check if truncation lost words
    word_ids = inputs.word_ids()
    max_word_id = max(wid for wid in word_ids if wid is not None)
    if max_word_id < len(words) - 1:
        return None  # sentence was truncated, skip it

    # Get subword-to-word alignment
    word_to_subword = align_subwords_to_words(tokenizer, words, inputs)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions: tuple of (1, num_heads, seq_len, seq_len) per layer
    num_layers = len(outputs.attentions)
    num_heads = outputs.attentions[0].shape[1]
    num_words = len(words)

    word_attention = np.zeros((num_layers, num_heads, num_words, num_words))

    for layer in range(num_layers):
        attn = outputs.attentions[layer][0].detach().numpy()  # (num_heads, seq_len, seq_len)
        for head in range(num_heads):
            word_attention[layer, head] = aggregate_attention(
                attn[head], word_to_subword, method=aggregation
            )

    return word_attention, word_to_subword


if __name__ == "__main__":
    from data_loader import load_conllu

    tokenizer, model = load_model("bert-base-uncased")
    sents = load_conllu("en", max_sentences=1)
    sent = sents[0]

    print(f"Sentence: {sent['text']}")
    print(f"Words: {sent['words']}")

    result = extract_attention(tokenizer, model, sent)
    if result:
        attention, mapping = result
        print(f"Attention shape: {attention.shape}")
        print(f"Word-to-subword mapping: {mapping}")
    else:
        print("Sentence was truncated, skipped.")
