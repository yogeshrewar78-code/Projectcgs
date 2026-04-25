"""
Load and parse Universal Dependencies CoNLL-U files.
Extracts sentences with gold dependency structures.
"""

import os
from conllu import parse
from typing import List, Dict, Optional


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Mapping of language codes to their UD treebank filenames
LANGUAGE_FILES = {
    "en": "en_ewt-ud-test.conllu",
    "en_train": "en_ewt-ud-train.conllu",
    "hi": "hi_hdtb-ud-test.conllu",
    "es": "es_gsd-ud-test.conllu",
    "fr": "fr_gsd-ud-test.conllu",
    "de": "de_gsd-ud-test.conllu",
    "zh": "zh_gsd-ud-test.conllu",
}


def load_conllu(language: str = "en", max_sentences: Optional[int] = None) -> List[Dict]:
    """
    Load sentences from a CoNLL-U file.

    Returns a list of dicts, each with:
        - 'text': the raw sentence string
        - 'words': list of word forms (excluding multiword tokens)
        - 'heads': list of head indices (0 = root, 1-indexed)
        - 'deprels': list of dependency relation labels
    """
    filename = LANGUAGE_FILES.get(language)
    if filename is None:
        raise ValueError(f"Unknown language '{language}'. Available: {list(LANGUAGE_FILES.keys())}")

    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        data = parse(f.read())

    sentences = []
    for sent in data:
        words = []
        heads = []
        deprels = []

        for token in sent:
            # Skip multiword tokens (id is a range like "1-2")
            if isinstance(token["id"], tuple):
                continue
            words.append(token["form"])
            heads.append(token["head"])
            deprels.append(token["deprel"])

        # Skip empty sentences
        if not words:
            continue

        sentences.append({
            "text": sent.metadata.get("text", " ".join(words)),
            "words": words,
            "heads": heads,   # 1-indexed, 0 = root
            "deprels": deprels,
        })

        if max_sentences and len(sentences) >= max_sentences:
            break

    return sentences


def get_dependency_pairs(sentence: Dict) -> List[Dict]:
    """
    Extract (dependent, head, relation) triples from a sentence.

    Returns list of dicts with:
        - 'dep_idx': 0-indexed position of the dependent word
        - 'head_idx': 0-indexed position of the head word (None if root)
        - 'deprel': the dependency relation label
        - 'dep_word': the dependent word form
        - 'head_word': the head word form (None if root)
    """
    pairs = []
    words = sentence["words"]
    heads = sentence["heads"]
    deprels = sentence["deprels"]

    for i, (word, head, deprel) in enumerate(zip(words, heads, deprels)):
        if head == 0:
            # Root — no head word
            pairs.append({
                "dep_idx": i,
                "head_idx": None,
                "deprel": deprel,
                "dep_word": word,
                "head_word": None,
            })
        else:
            head_idx = head - 1  # convert to 0-indexed
            pairs.append({
                "dep_idx": i,
                "head_idx": head_idx,
                "deprel": deprel,
                "dep_word": word,
                "head_word": words[head_idx],
            })

    return pairs


if __name__ == "__main__":
    # Quick test
    sents = load_conllu("en", max_sentences=3)
    for s in sents:
        print(f"\n{s['text']}")
        pairs = get_dependency_pairs(s)
        for p in pairs:
            arrow = f"{p['dep_word']} --{p['deprel']}--> {p['head_word']}"
            print(f"  {arrow}")
