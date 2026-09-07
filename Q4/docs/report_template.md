# Question 4 Report — Integrated Background Editor

## 1. System overview

Describe how Q1, Q3, the Q4 grammar subsystem, and Streamlit share one live text stream.

## 2. Reuse and integration

- Q1 English joint beam decoder: reused unchanged through `src/q1_reuse/`.
- Q3 spelling corrector: reused unchanged through `src/q3_reuse/`.
- Q1 parameters: max word length = 20, beam width = 5, alpha = 1.0, beta = 1.0.
- Q3 real-word margin = 1.5.

## 3. Q4 choices

- Merge probability `p = 0.08`.
- Grammar trigger interval `N = 10` words.
- Add-k smoothing `k = 0.1`.

Explain the speed/false-alert tradeoffs and report measured results.

## 4. PCFG and CKY

Explain Penn Treebank PCFG induction, CNF transformation, lexical/unary/binary rules, Viterbi chart cells, and graceful failure.

### Tagset reconciliation

Explain the Universal POS → Penn Treebank mapping in `src/tag_mapping.py`, including any information loss (e.g. Universal `VERB` collapsing multiple PTB verb forms).

## 5. Live alerts

Explain the exact order:

1. segmentation alert;
2. spelling alert;
3. trigger-level grammar and real-word check.

## 6. Final sentence comparison

Include the generated `results/final_sentence_analysis.csv` table and explain the method-selection rule.

## 7. Speed Demon

Report:

| Metric | Value |
|---|---:|
| Number of words | 1000 |
| Full segmentation + spelling | ... |
| Full average per word | ... |
| Grammar-only | ... |
| Grammar average | ... |
| Added segmentation + spelling | ... |

## 8. Comparative analysis

### Live vs final agreement

Give examples where the trigger-level alert agreed/disagreed with the final verdict.

### PCFG vs n-grams

Discuss structural vs local word-sequence judgments.

### N and p

Discuss alert latency and false-alert behavior.

### Subsystem interactions

Give at least two examples where segmentation or spelling changed PCFG parseability or the final method selection.

## 9. Screenshots / sample runs

Include at least two full runs on different random passages and a screenshot/transcript of Streamlit live deployment.
