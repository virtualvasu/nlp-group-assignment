# NLP Group Assignment

**Group 21:** Krishna Jhanwar (12341250), Siddharth Rai (12342050), Slok Tulsyan (12342080), Vasu Garg (12342330), Gaurav Kumar (12340790)

Each question's report is a LaTeX source (`.tex`) compiled to PDF, located alongside its code.

## Question 1 — Word Segmentation and POS Tagging

Located in [`Q_1/`](Q_1/):

- `main_q1.py` — entry point: runs segmentation + POS tagging end-to-end and prints results on the sample/test strings.
- `data_loader.py` — loads and splits the Brown (English) and UD Spanish-GSD corpora, extracting morphology features for Spanish.
- `ngram_models.py` — trigram language model (for segmentation) and HMM/MEMM POS tagger models.
- `decoders.py` — Viterbi segmentation and POS-tagging decoders (beam-pruned dynamic programming).
- `baselines.py` — greedy longest-match segmentation baseline and most-frequent-tag baseline.
- `evaluation.py` — accuracy, confusion matrix, and segmentation-vs-tagging error-source breakdown.
- `models/` — pretrained/pickled trigram LM, HMM, and MEMM taggers.
- `results/` — confusion matrix figures.
- Report: [`report.pdf`](Q_1/report.pdf) ([source](Q_1/report.tex))

## Question 2 — Transition-Based Dependency Parser

Located in [`Q_2/`](Q_2/):

- `question2_dependency_parser.ipynb` — full solution: CoNLL-U parsing, arc-standard oracle simulation, feature extraction, classifier training, parser inference, and LAS/UAS evaluation.
- `dataset-UD_English-EWT/` — UD_English-EWT train/dev data used for training and evaluation.
- `figures/` — figures used in the report.
- Report: [`report.pdf`](Q_2/report.pdf) ([source](Q_2/report.tex))


## Question 3 — Spelling Corrector

Located in [`Q_3/`](Q_3/):

- `main.py` — demo entry point (evaluation run + speed benchmark + live CLI).
- `src/data_prep.py` — Brown-corpus vocabulary, unigram frequencies, and bigram language model.
- `src/candidate_gen.py` — Method A (edit-distance-1) and Method B (symmetric-delete/SymSpell) candidate generation.
- `src/corrector.py` — non-word and real-word correction logic.
- `src/evaluate.py` — test-set generation, accuracy evaluation, and the "Speed Demon" 1,000-word benchmark.
- `src/cli.py` — continuous interactive terminal CLI with highlighting and latency reporting.
- `data/` — pretrained/pickled vocabulary, frequency, and bigram-model artifacts.
- Report: [`Q3__Spelling_Corrector_Report.pdf`](Q_3/Q3__Spelling_Corrector_Report.pdf) ([source](Q_3/Q3__Spelling_Corrector_Report.tex))

## Question 4 — Integrated Background Editor

Located in [`Q_4/`](Q_4/):

- `app.py` — Streamlit live-typing/deployment app (simulated typing + incremental live typing, alerts, final analysis).
- `main.py` / `run_submission.py` — batch/CLI entry points for the full pipeline and demo runs.
- `src/live_pipeline.py` — background segmentation/spelling/grammar alert pipeline.
- `src/pcfg_parser.py` — PCFG induction from the Penn Treebank sample and Viterbi/CKY most-probable-parse.
- `src/tag_mapping.py` — POS tagset reconciliation between Q1's tagger and PCFG's Penn Treebank tags.
- `src/ngram_q4.py` — shared smoothed bigram/trigram language models.
- `src/final_analysis.py` — end-of-passage per-sentence PCFG/bigram/trigram comparison and verdict.
- `src/benchmark.py` — Speed Demon (1,000-word) latency benchmark.
- `src/model_loader.py` — loads (without retraining) the reused Q1 and Q3 trained artifacts.
- `artifacts/` — reused pretrained Q1/Q3 model files (trigram LM, MEMM tagger, vocab/bigram data).
- `results/` — demo run transcripts, final sentence-analysis tables, and speed-benchmark output.
- Report: [`Q4_Final_Submission_Report.pdf`](Q_4/Q4_Final_Submission_Report.pdf) ([source](Q_4/Q4_Final_Submission_Report.tex))
