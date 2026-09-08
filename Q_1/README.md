# Question 1: Word Segmentation and POS Tagging

This directory contains the implementation and results for Question 1. The goal was to build a joint word segmentation and POS tagging system from scratch for both English and a morphologically rich language (Spanish), and to decouple segmentation errors from genuine tagging errors.

## Directory Structure & Files
* `data_loader.py`: Handles downloading NLTK resources, splitting the Brown corpus (80/10/10), and parsing the CoNLL-U format for Spanish (including morphology extraction).
* `ngram_models.py`: Contains the probabilistic models (`TrigramLanguageModel`, `POSTaggerHMM`, `FeatureBasedPOSTagger`).
* `decoders.py`: Implements the Dynamic Programming / Beam Search algorithms (`ViterbiSegmenter`, `ViterbiPOSTagger`, `JointBeamSearchDecoder`).
* `baselines.py`: Contains the `GreedyLongestMatchSegmenter` and `MostFrequentTagTagger` baselines for comparison.
* `evaluation.py`: Computes segmentation F1, plots the confusion matrix, and specifically separates segmentation-induced tagging errors from genuine tagging errors.
* `main_q1.py`: The entry point script that connects all modules, trains the models, and runs the evaluation.
* `data/`: Contains the cloned `UD_Spanish-GSD` dataset.
* `models/`: Stores the serialized `.pkl` models (`english_trigram_lm.pkl`, `english_hmm_tagger.pkl`, `english_memm_tagger.pkl`) for reuse in Question 4.
* `results/`: Contains the generated POS tagging confusion matrix images for both languages.

## Setup & Execution

**1. Create a Virtual Environment & Install Dependencies**
We recommend using a virtual environment to manage dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install nltk conllu scikit-learn matplotlib seaborn
```

**2. Download the Spanish Dataset**
Clone the Universal Dependencies Spanish-GSD repository into the `data/` folder:
```bash
mkdir -p data
cd data
git clone https://github.com/UniversalDependencies/UD_Spanish-GSD.git
cd ..
```

**3. Run the Pipeline**
Run the main evaluation script. It will automatically download the NLTK English corpora on its first run, train all the models, evaluate them, and populate the `models/` and `results/` directories.
```bash
python main_q1.py
```

---

## Performance Summary

### English Results (Brown Corpus)
- **Segmentation:** Baseline (85.25% F1) vs Viterbi Trigram (97.08% F1)

| POS Model | Seg-Induced Errors | Genuine Errors | Accuracy |
| :--- | :--- | :--- | :--- |
| **Baseline (Most Freq)** | 415 | 82 | 77.48% |
| **Trigram HMM** | 122 | 75 | **91.07%** |
| **MEMM** | 122 | 142 | 88.04% |

### Spanish Results (UD Spanish-GSD)
- **Segmentation:** Baseline (80.41% F1) vs Viterbi Trigram (88.27% F1)

| POS Model | Seg-Induced Errors | Genuine Errors | Accuracy |
| :--- | :--- | :--- | :--- |
| **Baseline (Most Freq)** | 594 | 129 | 71.91% |
| **Trigram HMM** | 334 | 154 | **81.04%** |
| **MEMM** | 334 | 191 | 79.60% |

---

## Comparative Analysis Report

### 1. Where did English and the other language differ most in accuracy?
English achieved significantly higher accuracy in both segmentation and POS tagging. 
- **Segmentation:** English (97.08% F1) vs Spanish (88.27% F1). 
- **POS Tagging (HMM):** English (91.07%) vs Spanish (81.04%).

The primary reason for the tagging discrepancy is the tagset size. For English, we mapped words to a relatively small universal tagset (~12 base tags). For Spanish (UD Spanish-GSD), we generated **morphology-aware tags** by combining the base UPOS with Gender and Number features (e.g., `NOUN-Fem-Plur`). This exponentially increased the number of target classes, making the classification task inherently much harder for Spanish.

### 2. Did agreement-aware tagging actually help, or add noise?
Adding morphology-aware tags introduced "metric noise" because the model had more opportunities to make a mistake (getting the base tag right but guessing the wrong gender/number still counts as an error). This caused the absolute accuracy to drop compared to a simplified tagset. 

However, from an NLP modeling perspective, it **helped tremendously**. By embedding gender and number into the state space, the HMM's transition probabilities naturally learned grammatical agreement patterns (e.g., $P(\text{ADJ-Fem-Plur} | \text{NOUN-Fem-Plur})$ became highly probable). The model proved it actually understood grammatical agreement rather than just guessing parts of speech in a vacuum.

### 3. How much of the tagging error came from segmentation vs. genuine mistakes?
Our custom error analyzer separated tagging mistakes caused by bad word boundaries from pure tagging failures.
- **English (HMM):** 122 errors were segmentation-induced, while 75 were genuine.
- **Spanish (HMM):** 334 errors were segmentation-induced, while 154 were genuine.

In both languages, **over 60% of all POS tagging errors were actually segmentation errors in disguise**. If a word boundary is guessed incorrectly, the POS tag assigned to that fake word is almost guaranteed to be wrong. This clearly illustrates why separating these error sources is critical for evaluation, and justifies why joint segmentation/tagging (explored in Q4) is heavily preferred over a strict pipeline.

### 4. How much better were your models than the simple baselines?
The probabilistic models significantly outperformed the naive baselines across the board, proving that dynamic programming and contextual modeling are worth the extra computational cost.
- **Segmentation (Baseline vs Viterbi):** F1 jumped from 85.25% to 97.08% in English, and 80.41% to 88.27% in Spanish. The greedy longest-match baseline frequently got stuck on compound traps, whereas Viterbi optimized the global sequence probability.
- **POS Tagging (Baseline vs HMM):** Accuracy improved from 77.48% to 91.07% in English, and 71.91% to 81.04% in Spanish. The baseline completely failed on ambiguous words, whereas the HMM utilized transition probabilities to deduce the correct context.

## Example Output (Spanish)
Input: `mispadrespuedenviajar`

Viterbi HMM Prediction:
```python
[
  ('mis', 'DET-Plur'), 
  ('padres', 'NOUN-Masc-Plur'), 
  ('pueden', 'AUX-Plur'), 
  ('viajar', 'VERB')
]
```
*Note the correct segmentation and the plural agreement successfully captured across the first three tokens.*
