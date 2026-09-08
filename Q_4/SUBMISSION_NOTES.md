# Q4 Submission Notes

## Final package contents

This ZIP contains the complete Q4 source tree:

- integrated Q1/Q3 reuse modules;
- Q1/Q3 artifact loader;
- incremental live pipeline;
- measured `[SEGMENT-ALERT]`, `[SPELL-ALERT]`, and `[GRAMMAR-ALERT]` timing;
- PCFG + custom CKY/Viterbi parser;
- Universal/Penn Treebank tag reconciliation, including `PRT -> RP`;
- shared smoothed n-gram analysis;
- final sentence comparison;
- Streamlit simulated/live UI;
- exact 1,000-word Speed Demon;
- tests and report template.

## Intentionally not included

The original Q1 serialized model files and Q3 Brown cache binaries are large/generated artifacts and are not duplicated in this source ZIP. Put the group's actual trained files into `artifacts/q1_models/` and `artifacts/q3_data/` using the paths documented in `README.md`.

`prepare_models.py` only validates their presence; it does not retrain Q1 or Q3.

## Final verification commands

```bash
pip install -r requirements.txt
python prepare_models.py
pytest -q
python run_submission.py
streamlit run app.py
```

## What must be added to the final report

Use the **actual** generated values/files from the submission machine:

- two full sample-run transcripts (`seed=42`, `seed=123`);
- final sentence-analysis CSVs;
- exact 1,000-word benchmark numbers;
- Streamlit screenshot/transcript;
- live-vs-final agreement examples;
- PCFG-vs-n-gram examples;
- `N=10` / `p=0.08` tradeoff discussion;
- at least two concrete subsystem interactions.

Do not fabricate timing numbers or screenshots.
