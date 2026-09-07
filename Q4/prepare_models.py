"""Validate the trained Q1/Q3 artifacts required by Q4.

Q4 must reuse the trained artifacts from Questions 1 and 3; it does not retrain
them. Copy the group's original artifacts into artifacts/q1_models and
artifacts/q3_data before running the submission.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    ROOT / 'artifacts' / 'q1_models' / 'english_trigram_lm.pkl',
    ROOT / 'artifacts' / 'q1_models' / 'english_memm_tagger.pkl',
    ROOT / 'artifacts' / 'q3_data' / 'vocab_freq.pkl',
    ROOT / 'artifacts' / 'q3_data' / 'bigram_model.pkl',
    ROOT / 'artifacts' / 'q3_data' / 'sentences.pkl',
]

if __name__ == '__main__':
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
    if missing:
        print('Missing required trained Q1/Q3 artifacts:')
        for p in missing:
            print('  -', p)
        print('\nCopy the original Q1/Q3 trained artifacts into these paths, then rerun.')
        raise SystemExit(1)
    print('All required Q1/Q3 trained artifacts are present. No retraining performed.')
