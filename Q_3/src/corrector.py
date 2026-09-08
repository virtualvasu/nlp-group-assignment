"""
Part 3: Spelling Correction Logic
===================================
Non-word errors  -> best candidate = highest unigram frequency, using
                     candidates pooled from BOTH Method A and Method B.
Real-word errors -> a vocabulary word may still be wrong in context
                     (e.g. "I ate an apply"). We generate candidates for
                     the word, then compare the bigram log-probability of
                     the phrase with the original word vs. each candidate,
                     and suggest a swap only if a candidate is
                     significantly more probable in context.
"""

import re

from src.candidate_gen import (
    build_symspell_dictionary,
    method_a_candidates,
    method_b_candidates,
)
from src.data_prep import phrase_log_prob

WORD_RE = re.compile(r"[a-zA-Z]+")

# How much more probable (in log space) a real-word candidate's phrase must
# be than the original phrase before we trust it enough to "correct" a
# perfectly valid word. Guards against over-correcting on noisy bigram
# counts.
REAL_WORD_LOG_MARGIN = 1.5


class SpellingCorrector:
    def __init__(self, vocab, freq, bigram_counts, unigram_counts, vocab_size):
        self.vocab = vocab
        self.freq = freq
        self.bigram_counts = bigram_counts
        self.unigram_counts = unigram_counts
        self.vocab_size = vocab_size
        self.delete_dict = build_symspell_dictionary(vocab)

    # -- candidate generation (pools both methods) --------------------------
    def candidates(self, word):
        a = method_a_candidates(word, self.vocab)
        b = method_b_candidates(word, self.delete_dict, self.vocab)
        return a | b

    def candidates_method_a(self, word):
        return method_a_candidates(word, self.vocab)

    def candidates_method_b(self, word):
        return method_b_candidates(word, self.delete_dict, self.vocab)

    # -- Part 3.1: non-word error correction ---------------------------------
    def correct_nonword(self, word):
        """`word` is assumed OOV. Returns (best_correction, changed: bool)."""
        lw = word.lower()
        if lw in self.vocab:
            return word, False  # not actually a non-word error

        cands = self.candidates(lw)
        if not cands:
            return word, False  # no correction found within edit distance 1

        best = max(cands, key=lambda w: self.freq.get(w, 0))
        return best, True

    # -- Part 3.2: real-word error correction --------------------------------
    def correct_realword(self, sentence_words, index):
        """`sentence_words` is a list of lower-cased tokens; `index` points
        at a word that IS in the vocabulary but might be contextually wrong.
        Returns (best_word_for_that_slot, changed: bool).
        """
        word = sentence_words[index]
        if word not in self.vocab:
            return word, False  # that's a non-word error, not this function's job

        cands = self.candidates(word)
        if not cands:
            return word, False

        def phrase_with(w):
            trial = list(sentence_words)
            trial[index] = w
            lo = max(0, index - 1)
            hi = min(len(trial), index + 2)
            return trial[lo:hi]

        original_lp = phrase_log_prob(
            phrase_with(word), self.bigram_counts, self.unigram_counts, self.vocab_size
        )

        best_word, best_lp = word, original_lp
        for c in cands:
            lp = phrase_log_prob(
                phrase_with(c), self.bigram_counts, self.unigram_counts, self.vocab_size
            )
            if lp > best_lp:
                best_word, best_lp = c, lp

        if best_word != word and (best_lp - original_lp) > REAL_WORD_LOG_MARGIN:
            return best_word, True
        return word, False

    # -- full-sentence correction (used by the CLI, Part 5) ------------------
    def correct_sentence(self, sentence):
        """Takes a raw sentence string, returns (corrected_string, list of
        (original_token, corrected_token) for every token that changed).
        Preserves original casing/punctuation as much as practical.
        """
        tokens = WORD_RE.findall(sentence)
        # Walk through with a simple tokenizer that keeps punctuation spans
        spans = list(WORD_RE.finditer(sentence))
        lowered = [m.group(0).lower() for m in spans]

        out_words = list(lowered)
        changes = []

        for i, w in enumerate(lowered):
            if w not in self.vocab:
                corrected, changed = self.correct_nonword(w)
            else:
                corrected, changed = self.correct_realword(lowered, i)
            if changed:
                out_words[i] = corrected
                changes.append((spans[i].group(0), corrected))
            else:
                out_words[i] = spans[i].group(0)  # keep original casing/text

        # Rebuild the sentence with original non-word spans (spacing, punctuation)
        result = []
        last_end = 0
        for m, new_word in zip(spans, out_words):
            result.append(sentence[last_end:m.start()])
            result.append(new_word)
            last_end = m.end()
        result.append(sentence[last_end:])
        return "".join(result), changes
