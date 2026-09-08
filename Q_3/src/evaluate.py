"""
Part 4: Evaluation and "Speed Demon" Benchmark
=================================================
1. Test-set generation: 10% of Brown corpus sentences, each contributing
   one non-word-error example and (where possible) one real-word-error
   example, built by introducing a single random edit into a random word.
2. Accuracy of the full corrector on both test sets.
3. Speed Demon benchmark: 1,000 misspelled words through Method A's
   candidate generator vs. Method B's, timed head-to-head.
"""

import json
import os
import random
import time

from src.candidate_gen import LETTERS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Single-edit corruption
# ---------------------------------------------------------------------------
def _random_single_edit(word, rng):
    """Apply ONE random edit operation (delete/insert/replace/transpose)
    to `word` and return the corrupted string. May occasionally return the
    same word (e.g. replacing a letter with itself) -- caller should retry.
    """
    if len(word) < 2:
        op = rng.choice(["insert", "replace"])
    else:
        op = rng.choice(["delete", "insert", "replace", "transpose"])

    i = rng.randrange(len(word))
    if op == "delete":
        return word[:i] + word[i + 1:]
    if op == "insert":
        c = rng.choice(LETTERS)
        return word[:i] + c + word[i:]
    if op == "replace":
        c = rng.choice(LETTERS)
        return word[:i] + c + word[i + 1:]
    if op == "transpose":
        if i == len(word) - 1:
            i -= 1
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word


def _make_nonword_error(word, vocab, rng, max_tries=25):
    """Corrupt `word` until the result is NOT a vocabulary word."""
    for _ in range(max_tries):
        cand = _random_single_edit(word, rng)
        if cand != word and cand not in vocab:
            return cand
    return None


def _make_realword_error(word, vocab, rng, max_tries=25):
    """Corrupt `word` until the result IS a *different* vocabulary word."""
    for _ in range(max_tries):
        cand = _random_single_edit(word, rng)
        if cand != word and cand in vocab:
            return cand
    return None


def generate_test_set(sentences, vocab, sample_fraction=0.1, seed=42, min_word_len=3):
    """Returns:
        nonword_examples: list of dicts {sentence, index, original, misspelled}
        realword_examples: list of dicts {sentence, index, original, misspelled}
    """
    rng = random.Random(seed)
    sample_size = max(1, int(len(sentences) * sample_fraction))
    sampled = rng.sample(sentences, sample_size)

    nonword_examples, realword_examples = [], []

    for sent in sampled:
        eligible_idx = [i for i, w in enumerate(sent) if len(w) >= min_word_len]
        if not eligible_idx:
            continue
        idx = rng.choice(eligible_idx)
        original = sent[idx]

        nw = _make_nonword_error(original, vocab, rng)
        if nw:
            nonword_examples.append(
                {"sentence": sent, "index": idx, "original": original, "misspelled": nw}
            )

        rw = _make_realword_error(original, vocab, rng)
        if rw:
            realword_examples.append(
                {"sentence": sent, "index": idx, "original": original, "misspelled": rw}
            )

    return nonword_examples, realword_examples


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------
def evaluate_nonword_accuracy(corrector, nonword_examples):
    correct = 0
    for ex in nonword_examples:
        predicted, _ = corrector.correct_nonword(ex["misspelled"])
        if predicted == ex["original"]:
            correct += 1
    total = len(nonword_examples)
    return correct / total if total else 0.0, correct, total


def evaluate_realword_accuracy(corrector, realword_examples):
    correct = 0
    for ex in realword_examples:
        corrupted_sentence = list(ex["sentence"])
        corrupted_sentence[ex["index"]] = ex["misspelled"]
        predicted, _ = corrector.correct_realword(corrupted_sentence, ex["index"])
        if predicted == ex["original"]:
            correct += 1
    total = len(realword_examples)
    return correct / total if total else 0.0, correct, total


# ---------------------------------------------------------------------------
# Speed Demon benchmark
# ---------------------------------------------------------------------------
def speed_demon_benchmark(corrector, nonword_examples, n=1000, seed=7):
    """Isolate non-word candidate generation: run the SAME batch of exactly
    `n` misspelled words through Method A and Method B, timing each.
    """
    rng = random.Random(seed)
    pool = [ex["misspelled"] for ex in nonword_examples]

    # Top up the pool if the natural test set has fewer than n examples, by
    # corrupting random vocab words, so the benchmark batch is always
    # exactly `n` words regardless of how many organic errors were found.
    if len(pool) < n:
        vocab_list = list(corrector.vocab)
        while len(pool) < n:
            w = rng.choice(vocab_list)
            corrupted = _make_nonword_error(w, corrector.vocab, rng)
            if corrupted:
                pool.append(corrupted)
    batch = pool[:n]

    start_a = time.perf_counter()
    for w in batch:
        corrector.candidates_method_a(w)
    time_a = time.perf_counter() - start_a

    start_b = time.perf_counter()
    for w in batch:
        corrector.candidates_method_b(w)
    time_b = time.perf_counter() - start_b

    return {
        "n_words": n,
        "method_a_seconds": time_a,
        "method_b_seconds": time_b,
        "speedup_factor": (time_a / time_b) if time_b > 0 else float("inf"),
    }


CONCLUSION_TEMPLATE = """\
Speed Demon Benchmark conclusion
---------------------------------
Method A (brute-force edit distance 1) generates every possible string
reachable by one delete, transpose, replace, or insert. For a word of
length n over a 26-letter alphabet that is roughly (26n + 25) candidate
strings, and EVERY one of them is then checked for vocabulary membership.
The cost is dominated by that O(26n) string construction + hashing, per
word.

Method B (Symmetric Delete) only ever performs n *deletions* of the query
word (no insertions/replacements at query time) and looks each of those
n+1 short strings up directly in a pre-built hash map. Its per-query cost
is O(n) string construction + O(n) hash lookups -- no factor of 26 -- with
all the combinatorial work of relating deletions back to full words paid
ONCE, offline, while building the dictionary.

On this run, {n_words} misspelled words took {method_a_seconds:.4f}s with
Method A and {method_b_seconds:.4f}s with Method B, i.e. Method B was
~{speedup_factor:.1f}x faster. This matches the theoretical expectation:
removing the factor-of-26 alphabet loop from the per-query hot path is
exactly why Method B scales so much better, at the one-time cost of
building a larger pre-processing dictionary.
"""


def run_full_evaluation(corrector, sentences, vocab):
    print("[evaluate] Generating test set (10% of Brown sentences)...")
    nonword_examples, realword_examples = generate_test_set(sentences, vocab)
    print(f"[evaluate] Non-word examples: {len(nonword_examples)} | Real-word examples: {len(realword_examples)}")

    nw_acc, nw_correct, nw_total = evaluate_nonword_accuracy(corrector, nonword_examples)
    rw_acc, rw_correct, rw_total = evaluate_realword_accuracy(corrector, realword_examples)
    print(f"[evaluate] Non-word accuracy: {nw_acc:.2%} ({nw_correct}/{nw_total})")
    print(f"[evaluate] Real-word accuracy: {rw_acc:.2%} ({rw_correct}/{rw_total})")

    print("[evaluate] Running Speed Demon benchmark on 1,000 words...")
    speed_results = speed_demon_benchmark(corrector, nonword_examples, n=1000)
    conclusion = CONCLUSION_TEMPLATE.format(**speed_results)
    print(conclusion)

    report = {
        "nonword_accuracy": nw_acc,
        "nonword_correct": nw_correct,
        "nonword_total": nw_total,
        "realword_accuracy": rw_acc,
        "realword_correct": rw_correct,
        "realword_total": rw_total,
        "speed_demon": speed_results,
    }
    with open(os.path.join(RESULTS_DIR, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "speed_demon_conclusion.txt"), "w") as f:
        f.write(conclusion)

    print(f"[evaluate] Report written to {RESULTS_DIR}/evaluation_report.json")
    return report
