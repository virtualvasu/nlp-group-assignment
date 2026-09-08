# Spelling Corrector — NLP Group Assignment (Question 3)

A spelling corrector built on the NLTK **Brown corpus** that:

- corrects **non-word errors** (misspellings not in the vocabulary) using unigram frequency,
- corrects **real-word errors** (a valid word used wrongly, e.g. *"I would like to **sea** the world"*) using a bigram context model,
- compares **two candidate-generation strategies** — brute-force edit-distance-1 vs. Symmetric Delete — in a head-to-head speed benchmark,
- and ships as a live, interactive terminal app.

Everything below maps 1:1 onto the assignment's Parts 1–5.

---

## Project structure

```
spelling-corrector/
├── main.py                    # Entry point: evaluate / cli / demo
├── requirements.txt
├── README.md
├── src/
│   ├── data_prep.py           # Part 1: vocab, unigram freq, bigram model
│   ├── candidate_gen.py       # Part 2: Method A (brute-force) & Method B (SymSpell)
│   ├── corrector.py           # Part 3: non-word & real-word correction logic
│   ├── evaluate.py            # Part 4: test-set generation, accuracy, Speed Demon benchmark
│   └── cli.py                 # Part 5: interactive terminal CLI
├── data/                      # Auto-created: cached corpus/vocab/bigram artifacts (.pkl)
└── results/                   # Auto-created: evaluation_report.json, speed_demon_conclusion.txt
```

`data/` and `results/` are generated the first time you run the code — nothing needs to be downloaded or prepared by hand beyond `pip install -r requirements.txt`. The Brown corpus itself is fetched automatically via `nltk.download('brown')` on first run and cached by NLTK.

---

## Setup

```bash
cd spelling-corrector
pip install -r requirements.txt
```

No API keys or external services are needed — everything runs locally against the Brown corpus.

## How to run

### 1. Run everything (evaluation + demo) — the default

```bash
python main.py
# or explicitly:
python main.py all
```

This will (in order):
1. Build/cache the vocabulary, unigram frequencies, and bigram model from Brown (Part 1).
2. Generate the 10%-of-sentences test set and report non-word / real-word accuracy (Part 4).
3. Run the Speed Demon benchmark (1,000 words, Method A vs. Method B) and print the conclusion (Part 4).
4. Run the corrector on the four example sentences from the assignment brief (sanity demo).

### 2. Just the evaluation / benchmark

```bash
python main.py evaluate
```

Writes `results/evaluation_report.json` (accuracies + timings) and `results/speed_demon_conclusion.txt`.

### 3. Just the demo sentences

```bash
python main.py demo
```

### 4. The interactive CLI (Part 5)

```bash
python main.py cli
```

```
> I hav a good feeling about this.
Corrected: I **had** a good feeling about this.
Changed:   hav -> had
Latency:   1.45 ms

> exit
Goodbye!
```

Changed words are wrapped in `**asterisks**` and, on ANSI-capable terminals, also shown in bold green. Type `exit` at any point to quit.

---

## Design notes, part by part

### Part 1 — Corpus & Model Preparation (`src/data_prep.py`)
- Sentences are pulled from `nltk.corpus.brown.sents()`, lower-cased, and filtered to alphabetic tokens only (numbers/punctuation carry no meaningful "edit distance" notion for a speller).
- **Unigram model**: a `collections.Counter` over all tokens → vocabulary = its key set, frequency = its values.
- **Bigram model**: counts of `(w1, w2)` pairs with sentence boundaries marked as `<s>` / `</s>`, plus **add-one (Laplace) smoothing** so unseen bigrams get a small non-zero probability instead of killing the whole phrase probability. Probabilities are combined in **log-space** to avoid underflow over longer phrases.
- All three artifacts are pickled into `data/` so every subsequent run is instant.

### Part 2 — Candidate Generation (`src/candidate_gen.py`)
- **Method A** (`edits1`): the classic brute-force generator — every deletion, transposition, replacement, and insertion of a word, ~`26n + 25` strings for a word of length `n`, filtered down to real vocabulary words.
- **Method B** (Symmetric Delete): a *pre-processing* pass deletes one character from every vocabulary word and maps each result back to its source word(s) (`build_symspell_dictionary`). At query time we only delete characters from the *misspelled* word (never insert or substitute) and do direct hash-map lookups — `O(n)` instead of `O(26n)`. A cheap true-edit-distance check filters the rare distance-2 collision that delete-delete matching can surface, so results stay correct as well as fast.

### Part 3 — Correction Logic (`src/corrector.py`)
- **Non-word errors**: candidates are pooled from *both* methods, and the highest-unigram-frequency candidate wins.
- **Real-word errors**: candidates are generated for a word that's already valid; we compute the bigram log-probability of the local phrase (previous word, target, next word) with the original word vs. each candidate, and only substitute if a candidate's phrase is *significantly* more probable (`REAL_WORD_LOG_MARGIN`) — this keeps the corrector from "fixing" words that are already fine just because of sparse bigram counts.
- `correct_sentence()` ties both together for full-sentence correction, preserving original casing/punctuation of unchanged tokens.

### Part 4 — Evaluation (`src/evaluate.py`)
- **Test set**: 10% of Brown sentences are sampled; for each, one word is picked and a single random edit (delete/insert/replace/transpose) is applied — retried until the result is out-of-vocabulary (→ non-word example) and, separately, until it lands on a *different* real vocabulary word (→ real-word example).
- **Accuracy**: fraction of test examples where the corrector recovers the original word.
- **Speed Demon benchmark**: exactly 1,000 misspelled words are passed through Method A's and Method B's candidate generators (candidate generation only, isolated from frequency/bigram scoring), timed with `time.perf_counter()`. The conclusion is auto-written to `results/speed_demon_conclusion.txt`, explaining the O(26n) vs. O(n) difference.

### Part 5 — Live CLI (`src/cli.py`)
- A plain `while True` loop: read a line, correct it, print the corrected sentence with changed words in `**asterisks**` (+ ANSI bold-green where supported), print latency in milliseconds, exit on `exit`.

---

## Sample results

From a full run (`python main.py all`) on the Brown corpus (~40k vocabulary words, ~57k sentences):

| Metric | Value |
|---|---|
| Non-word correction accuracy | ~84% |
| Real-word correction accuracy | ~67% |
| Method A (1,000 words) | ~0.065 s |
| Method B (1,000 words) | ~0.010 s |
| Speedup | ~6x |

Exact numbers vary slightly run to run only if you change the random seed in `generate_test_set` / `speed_demon_benchmark` (both are seeded by default for reproducibility). Full numbers are always written to `results/evaluation_report.json`.

**Why Method B wins:** Method A re-derives every possible edit of a word from scratch at query time — a `26×n` alphabet sweep. Method B pushes that combinatorial work into a one-time, offline pre-processing step over the vocabulary, so each query only needs `n` deletions and `n` hash-map lookups. The gap grows for longer words and larger batches, which is exactly what the benchmark is designed to expose.

---

## Notes / possible extensions
- The real-word margin (`REAL_WORD_LOG_MARGIN` in `corrector.py`) and the smoothing scheme (currently add-one) are the two easiest levers to tune if you want more/less aggressive corrections or a stronger language model (e.g. Kneser-Ney smoothing, trigrams).
- Currently only single-token edits are modeled (as scoped by the assignment: "within an edit distance of 1"); multi-error words are out of scope by design.
