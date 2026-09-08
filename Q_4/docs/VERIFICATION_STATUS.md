# Verification status for this checkout

## Code-level verification

The final package includes automated tests for:

- Python import/compile smoke coverage.
- Merged-token generation.
- OOV spelling correction and `[SPELL-ALERT]` timing.
- Final-analysis sentence-boundary behavior (the Q1 joint decoder is not run over a whole finalized sentence).
- Universal `PRT -> RP` Penn Treebank tag reconciliation.
- PCFG/CKY behavior when the Treebank corpus is available.

Run:

```bash
pytest -q
```

A PCFG test may be skipped on machines without the NLTK Penn Treebank corpus.

## Empirical verification to run on the submission machine

The final empirical values must come from the group's trained Q1/Q3 artifacts and the machine used for the demonstration. Run:

```bash
python prepare_models.py
python run_submission.py
streamlit run app.py
```

`run_submission.py` produces two deterministic sample transcripts (`seed=42` and `seed=123`) plus the standalone exact-1,000-word Speed Demon output under `results/`.

Do **not** fabricate benchmark timings or Streamlit screenshots. Copy the actual numbers into `docs/report_template.md` after running the final package.
