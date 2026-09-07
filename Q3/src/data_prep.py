"""
Part 1: Corpus and Model Preparation
=====================================
Builds, from the NLTK Brown corpus:
  - a vocabulary of unique words
  - a unigram frequency distribution
  - a bigram probability model (with add-one / Laplace smoothing)

Artifacts are cached to disk (data/*.pkl) so subsequent runs are instant.
"""

import os
import pickle
import re
from collections import Counter

import nltk

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

VOCAB_PATH = os.path.join(DATA_DIR, "vocab_freq.pkl")
BIGRAM_PATH = os.path.join(DATA_DIR, "bigram_model.pkl")
SENTENCES_PATH = os.path.join(DATA_DIR, "sentences.pkl")

WORD_RE = re.compile(r"^[a-z]+$")


def _ensure_brown_downloaded():
    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown")


def load_brown_sentences():
    """Return the Brown corpus as a list of sentences, each a list of
    lower-cased, alphabetic-only tokens. Non-alphabetic tokens (numbers,
    punctuation) are dropped since a spelling corrector for alphabetic
    words has no meaningful notion of an 'edit distance 1' correction
    for them.
    """
    _ensure_brown_downloaded()
    from nltk.corpus import brown

    sentences = []
    for sent in brown.sents():
        cleaned = [w.lower() for w in sent if WORD_RE.match(w.lower())]
        if cleaned:
            sentences.append(cleaned)
    return sentences


def build_vocab_and_freq(sentences):
    """Part 1.1: unique-word vocabulary + unigram frequency distribution."""
    freq = Counter()
    for sent in sentences:
        freq.update(sent)
    vocab = set(freq.keys())
    return vocab, freq


def build_bigram_model(sentences, vocab):
    """Part 1.2: bigram probability model P(w2 | w1) with add-one smoothing.

    Returns:
        bigram_counts: Counter of (w1, w2) -> count
        unigram_counts_for_bigram: Counter of w1 -> total count (context count)
        vocab_size: |V| used for Laplace smoothing denominator
    """
    bigram_counts = Counter()
    unigram_counts = Counter()
    for sent in sentences:
        padded = ["<s>"] + sent + ["</s>"]
        for w1, w2 in zip(padded, padded[1:]):
            bigram_counts[(w1, w2)] += 1
            unigram_counts[w1] += 1
    vocab_size = len(vocab) + 2  # + <s> and </s> pseudo-tokens
    return bigram_counts, unigram_counts, vocab_size


def bigram_prob(w1, w2, bigram_counts, unigram_counts, vocab_size):
    """Add-one (Laplace) smoothed P(w2 | w1)."""
    numerator = bigram_counts.get((w1, w2), 0) + 1
    denominator = unigram_counts.get(w1, 0) + vocab_size
    return numerator / denominator


def phrase_log_prob(words, bigram_counts, unigram_counts, vocab_size):
    """Log-probability of a short phrase (list of words) under the bigram
    model. Using log-space avoids underflow and makes comparisons additive.
    """
    import math

    padded = ["<s>"] + words + ["</s>"]
    log_p = 0.0
    for w1, w2 in zip(padded, padded[1:]):
        p = bigram_prob(w1, w2, bigram_counts, unigram_counts, vocab_size)
        log_p += math.log(p)
    return log_p


def build_and_cache_all(force_rebuild=False):
    """Build (or load cached) vocab/freq/bigram artifacts. Returns a dict
    bundling everything Part 2/3/4/5 need.
    """
    if (
        not force_rebuild
        and os.path.exists(VOCAB_PATH)
        and os.path.exists(BIGRAM_PATH)
        and os.path.exists(SENTENCES_PATH)
    ):
        with open(VOCAB_PATH, "rb") as f:
            vocab, freq = pickle.load(f)
        with open(BIGRAM_PATH, "rb") as f:
            bigram_counts, unigram_counts, vocab_size = pickle.load(f)
        with open(SENTENCES_PATH, "rb") as f:
            sentences = pickle.load(f)
        print(f"[data_prep] Loaded cached artifacts. |V|={len(vocab)}, sentences={len(sentences)}")
        return {
            "sentences": sentences,
            "vocab": vocab,
            "freq": freq,
            "bigram_counts": bigram_counts,
            "unigram_counts": unigram_counts,
            "vocab_size": vocab_size,
        }

    print("[data_prep] Building corpus artifacts from the Brown corpus (first run)...")
    sentences = load_brown_sentences()
    vocab, freq = build_vocab_and_freq(sentences)
    bigram_counts, unigram_counts, vocab_size = build_bigram_model(sentences, vocab)

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump((vocab, freq), f)
    with open(BIGRAM_PATH, "wb") as f:
        pickle.dump((bigram_counts, unigram_counts, vocab_size), f)
    with open(SENTENCES_PATH, "wb") as f:
        pickle.dump(sentences, f)

    print(f"[data_prep] Done. |V|={len(vocab)}, sentences={len(sentences)}")
    return {
        "sentences": sentences,
        "vocab": vocab,
        "freq": freq,
        "bigram_counts": bigram_counts,
        "unigram_counts": unigram_counts,
        "vocab_size": vocab_size,
    }


if __name__ == "__main__":
    data = build_and_cache_all()
    print("Sample vocab words:", list(data["vocab"])[:10])
    print("Most common words:", data["freq"].most_common(10))
