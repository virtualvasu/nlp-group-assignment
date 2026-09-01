# NLP Group Assignment

## Question 2 — Transition-Based Dependency Parser

Located in [`ques2/`](ques2/):

- `question2_dependency_parser.ipynb` — full solution: CoNLL-U parsing, arc-standard oracle simulation, feature extraction, classifier training, parser inference, and LAS/UAS evaluation.
- `report.pdf` — short report on design choices and results.
- `dataset-UD_English-EWT/` — UD_English-EWT train/dev data used for training and evaluation.
- `figures/` — figures used in the report.

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install jupyter notebook ipykernel scikit-learn numpy pandas
```

Then open `ques2/question2_dependency_parser.ipynb` and run all cells.
