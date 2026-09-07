# Question 4 — Integrated Background Editor

This directory implements **Q4** as one integrated live text editor rather than three separate demos.

## What is integrated

1. **Q1:** the group's trained English `JointBeamSearchDecoder` for suspicious/merged-token segmentation + POS tagging.
2. **Q3:** the group's trained vocabulary/unigram/bigram data and spelling corrector/candidate generators for non-word and trigger-level real-word correction.
3. **Q4:** PCFG induction from Penn Treebank, custom Viterbi CKY parsing, shared smoothed n-gram sentence analysis, live alerts, final comparison, Streamlit UI, and the exact 1,000-word Speed Demon.

The live stream is processed in this order:

`input token → Q1 segmentation (if suspicious) → Q3 non-word spelling → shared stream → every N words: grammar + Q3 real-word check → final PCFG/n-gram analysis`

## Important reuse rule

Q4 **does not retrain Q1 or Q3**. Their reusable source is vendored under `src/q1_reuse/` and `src/q3_reuse/`, while their trained artifacts must be supplied in `artifacts/`.

Required artifacts:

```text
artifacts/
├── q1_models/
│   ├── english_trigram_lm.pkl
│   └── english_memm_tagger.pkl
└── q3_data/
    ├── vocab_freq.pkl
    ├── bigram_model.pkl
    └── sentences.pkl
```

`prepare_models.py` only validates these files. It deliberately does **not** retrain Q1/Q3.

## Installation

```bash
cd Q4
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first run requires the assignment corpora: **Brown** and **Penn Treebank**. The passage sampler may also use Gutenberg/Reuters depending on the installed data. NLTK download prompts are handled by the relevant modules when possible.

## Run the final package

First place the real Q1/Q3 trained artifacts in the paths above:

```bash
python prepare_models.py
```

Then run the two required deterministic sample passages and the standalone benchmark:

```bash
python run_submission.py
```

This writes:

```text
results/
├── demo_seed_42.txt
├── demo_seed_123.txt
├── final_sentence_analysis_seed_42.csv
├── final_sentence_analysis_seed_123.csv
├── speed_demon.txt
└── q4_speed_demon.json
```

For quick debugging without the expensive benchmark:

```bash
python main.py demo --seed 42 --no-benchmark
```

Standalone benchmark:

```bash
python main.py benchmark
```

## Streamlit

```bash
streamlit run app.py
```

There are two modes.

### Simulated typing

A random 5–8 sentence passage is streamed word by word. Adjacent words are merged with probability `p=0.08`, creating realistic opportunities for Q1 segmentation. The UI shows the evolving text and measured alert latency.

### Live typing

The text area acts as the live editor. A completed whitespace-delimited token is processed once; an unfinished final token is held until a space/newline is entered. Processed tokens and alert history are kept in `st.session_state`, so Streamlit reruns do not repeatedly process old tokens. If already-processed text is edited/deleted, the pipeline is reset and rebuilt from the current prefix.

For clean sentence-level live final analysis, enter **one sentence per line**.

The UI reports:

- `[SEGMENT-ALERT]` with measured Q1-layer latency;
- `[SPELL-ALERT]` with measured Q3-layer latency;
- `[GRAMMAR-ALERT]` with measured trigger latency;
- separate live token and grammar-trigger averages;
- final PCFG/n-gram analysis;
- the exact 1,000-word Speed Demon.

## Latency accounting

The Q4 implementation measures the two assignment-requested layers separately:

- **Per-token segmentation + spelling:** timer starts before suspicious-token detection and ends after Q3 non-word correction, before the grammar trigger.
- **Per-trigger grammar + real-word:** measured independently inside the trigger checker.

Therefore a grammar alert no longer displays the old default `0.00 ms`; its displayed value is the actual trigger measurement. Segmentation/spelling alerts similarly show the measured per-token latency.

## Q4 settings

| Parameter | Value | Purpose |
|---|---:|---|
| Merge probability `p` | 0.08 | Exercise Q1 segmentation without merging most words. |
| Grammar trigger `N` | 10 words | Periodic local grammar/real-word check. |
| Q4 add-k `k` | 0.1 | Smoothing for Q4 bigram scoring. |
| Q1 max word length | 20 | Reused Q1 decoder setting. |
| Q1 beam width | 5 | Reused Q1 decoder setting. |
| Q1 alpha | 1.0 | Reused Q1 decoder setting. |
| Q1 beta | 1.0 | Reused Q1 decoder setting. |
| Q3 real-word margin | 1.5 | Reused Q3 decision threshold. |

The report should discuss how changing `p` affects segmentation-alert frequency and how changing `N` trades earlier grammar feedback against computation/false alerts.

## PCFG / CKY

The parser trains a PCFG from NLTK Penn Treebank trees after Chomsky Normal Form conversion. Parsing is a custom Viterbi CKY chart with lexical, unary, and binary productions. Unparseable sentences return a structured failure rather than crashing.

Q1's Universal/Brown POS tags are reconciled to Penn Treebank tags in `src/tag_mapping.py`. The mapping includes both `PART -> RP` and legacy `PRT -> RP`; this is important because PTB distinguishes categories more finely than the Universal tagset.

## Final sentence decision

For each finalized sentence:

1. obtain POS tags using Q1's trained POS model while preserving the finalized token boundaries;
2. attempt the PCFG parse;
3. compute shared bigram/trigram sentence scores;
4. prefer PCFG when it parses and is not a strong normalized-score outlier;
5. otherwise use trigram, then bigram as fallback;
6. emit a model-based grammaticality verdict.

The Q1 **joint segmentation decoder is not run on an entire finalized sentence with spaces removed**. It is reserved for the live suspicious-token segmentation step.

## Speed Demon

The benchmark creates **exactly 1,000 corrupted/OOV simulated words** and measures:

1. the full per-token Q1 segmentation + Q3 spelling layer with grammar triggers disabled;
2. the grammar/real-word trigger check in isolation on the same 1,000-token batch.

The report must present the actual measured total and average times. If Q1 OOV decoding is expensive, that is a legitimate finding: discuss throttling/optimization rather than changing or hiding the measurement.

## Tests

```bash
pytest -q
```

The test suite covers merged-token generation, OOV spelling alerting/timing, final-analysis boundary preservation, and `PRT -> RP` mapping. The PCFG test is skipped if Penn Treebank is unavailable.

## Report

Use `docs/report_template.md`. The report should contain actual results from `run_submission.py`, two full sample runs on different seeds, Streamlit evidence, the 1,000-word benchmark, live-vs-final discussion, PCFG-vs-n-gram discussion, `N/p` tradeoffs, and subsystem interaction examples.
