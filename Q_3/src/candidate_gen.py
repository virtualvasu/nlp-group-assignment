"""
Part 2: Candidate Generation Methods
=====================================
Method A: Standard brute-force generation of every string at edit
          distance 1 (deletes, transposes, replaces, inserts), then
          filtered against the vocabulary.

Method B: Symmetric Delete Spelling Correction (SymSpell-style). A
          pre-processing pass builds a dictionary mapping every 1-character
          deletion of every vocabulary word back to that word. At query
          time we only ever *delete* characters from the misspelled word
          (never insert/replace), which is O(len(word)) instead of
          O(len(word) * 26), and look the results up in a hash map.
"""

from collections import defaultdict

LETTERS = "abcdefghijklmnopqrstuvwxyz"


# ---------------------------------------------------------------------------
# Method A: standard edit-distance-1 generation
# ---------------------------------------------------------------------------
def edits1(word):
    """Return the set of ALL strings at edit distance 1 from `word`
    (deletions, transpositions, replacements, insertions). This is the
    classic brute-force generator: for a word of length n it produces
    O(26n) candidate strings, most of which are not real words.
    """
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in LETTERS]
    inserts = [L + c + R for L, R in splits for c in LETTERS]
    return set(deletes + transposes + replaces + inserts)


def method_a_candidates(word, vocab):
    """Generate edit-distance-1 strings, then keep only real vocabulary
    words. This is the step the assignment calls 'Method A: Standard Edit
    Distance 1 Generation' feeding into Part 3's frequency-based ranking.
    """
    raw = edits1(word)
    return {w for w in raw if w in vocab}


# ---------------------------------------------------------------------------
# Method B: Symmetric Delete Spelling Correction
# ---------------------------------------------------------------------------
def _deletes_of(word):
    """All strings obtained by deleting exactly one character from `word`."""
    if len(word) == 0:
        return set()
    return {word[:i] + word[i + 1:] for i in range(len(word))}


def build_symspell_dictionary(vocab):
    """Pre-processing step (done ONCE, offline): for every vocabulary word,
    map each of its 1-character deletions -> list of original words that
    produced it. We also map the word to itself so an exact match is O(1).

    Returns a dict: variant_string -> list[original_vocab_word]
    """
    delete_dict = defaultdict(list)
    for w in vocab:
        delete_dict[w].append(w)  # identity entry (edit distance 0)
        for d in _deletes_of(w):
            delete_dict[d].append(w)
    return delete_dict


def _levenshtein_le1(a, b):
    """Cheap true edit-distance-<=1 check, used only to verify the small
    candidate set SymSpell proposes (guards against the rare distance-2
    collisions that a pure delete-delete match can produce).
    """
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    edited = False
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        if edited:
            return False
        edited = True
        if la == lb:
            i += 1
            j += 1  # substitution
        else:
            j += 1  # insertion into `a` / deletion from `b`
    return True


def method_b_candidates(word, delete_dict, vocab):
    """Candidate Generation Step: generate the misspelled word's own
    1-character deletions (plus itself), and look each one up directly in
    the pre-processed delete_dict. No alphabet loop is needed at query
    time -- this is what makes Method B fast.
    """
    lookups = _deletes_of(word) | {word}
    found = set()
    for variant in lookups:
        for candidate in delete_dict.get(variant, ()):
            found.add(candidate)
    # Verify true edit distance <= 1 (delete-delete matching can, rarely,
    # surface a distance-2 word; this filter is O(len(candidates)) and
    # keeps Method B both fast and correct).
    return {c for c in found if c != word and _levenshtein_le1(word, c)}
