import time
import sys
import os

from data_loader import load_brown_corpus, get_spanish_data
from ngram_models import TrigramLanguageModel, POSTaggerHMM, FeatureBasedPOSTagger
from decoders import ViterbiSegmenter, ViterbiPOSTagger, JointBeamSearchDecoder
from baselines import GreedyLongestMatchSegmenter, MostFrequentTagTagger
from evaluation import compute_segmentation_accuracy, analyze_errors, plot_confusion_matrix

def train_and_evaluate(language, train_data, test_data, is_english=False):
    print(f"\n{'='*50}\nEvaluating {language}\n{'='*50}")
    
    # 1. Prepare data
    # sentences: list of words
    # tagged_sentences: list of (word, tag)
    train_sents = [[w for w, t in s] for s in train_data]
    test_sents = [[w for w, t in s] for s in test_data]
    
    # 2. Train Models
    print("Training Trigram Language Model...")
    lm = TrigramLanguageModel(k=0.1)
    lm.train(train_sents)
    
    print("Training HMM POS Tagger...")
    hmm = POSTaggerHMM(k=0.1)
    hmm.train(train_data)
    
    print("Training MEMM POS Tagger...")
    memm = FeatureBasedPOSTagger()
    memm.train(train_data)
    
    # Save models if English (for Q4)
    if is_english:
        print("Saving English models for Q4...")
        os.makedirs("models", exist_ok=True)
        lm.save("models/english_trigram_lm.pkl")
        hmm.save("models/english_hmm_tagger.pkl")
        memm.save("models/english_memm_tagger.pkl")
        
    # 3. Baselines
    print("Training Baselines...")
    greedy_seg = GreedyLongestMatchSegmenter(lm.vocab)
    mf_tagger = MostFrequentTagTagger()
    mf_tagger.train(train_data)
    
    # 4. Decoders
    viterbi_seg = ViterbiSegmenter(lm)
    viterbi_hmm = ViterbiPOSTagger(hmm)
    viterbi_memm = ViterbiPOSTagger(memm)
    joint_decoder = JointBeamSearchDecoder(lm, memm, beam_width=5)
    
    # 5. Evaluate on a subset of test data (e.g. 100 sentences) to save time
    test_subset = test_data[:100]
    
    seg_f1_baseline = 0
    seg_f1_viterbi = 0
    
    hmm_seg_errors, hmm_gen_errors, hmm_total = 0, 0, 0
    memm_seg_errors, memm_gen_errors, memm_total = 0, 0, 0
    base_seg_errors, base_gen_errors, base_total = 0, 0, 0
    
    all_actual_tags_hmm = []
    all_pred_tags_hmm = []
    
    print("Running evaluation on test subset...")
    for idx, gold_tagged_sent in enumerate(test_subset):
        if idx % 20 == 0:
            print(f"Processed {idx}/{len(test_subset)} sentences")
            
        gold_words = [w for w, t in gold_tagged_sent]
        text_no_spaces = "".join(gold_words)
        
        # --- Segmentation ---
        pred_words_baseline = greedy_seg.segment(text_no_spaces)
        pred_words_viterbi = viterbi_seg.segment(text_no_spaces)
        
        seg_f1_baseline += compute_segmentation_accuracy(pred_words_baseline, gold_words)
        seg_f1_viterbi += compute_segmentation_accuracy(pred_words_viterbi, gold_words)
        
        # --- Tagging ---
        pred_tags_baseline = mf_tagger.tag(pred_words_baseline)
        pred_tags_hmm = viterbi_hmm.tag(pred_words_viterbi)
        pred_tags_memm = viterbi_memm.tag(pred_words_viterbi)
        
        # --- Error Analysis ---
        se_b, ge_b, t_b, _, _ = analyze_errors(pred_tags_baseline, gold_tagged_sent)
        base_seg_errors += se_b; base_gen_errors += ge_b; base_total += t_b
        
        se_h, ge_h, t_h, a_t, p_t = analyze_errors(pred_tags_hmm, gold_tagged_sent)
        hmm_seg_errors += se_h; hmm_gen_errors += ge_h; hmm_total += t_h
        all_actual_tags_hmm.extend(a_t)
        all_pred_tags_hmm.extend(p_t)
        
        se_m, ge_m, t_m, _, _ = analyze_errors(pred_tags_memm, gold_tagged_sent)
        memm_seg_errors += se_m; memm_gen_errors += ge_m; memm_total += t_m

    n = len(test_subset)
    print(f"\n--- {language} Results ---")
    print(f"Segmentation F1 - Baseline: {seg_f1_baseline/n:.4f}, Viterbi: {seg_f1_viterbi/n:.4f}")
    
    print(f"\nPOS Tagging Errors (Baseline) - Seg-induced: {base_seg_errors}, Genuine: {base_gen_errors}, Accuracy: {1 - (base_seg_errors+base_gen_errors)/max(1, base_total):.4f}")
    print(f"POS Tagging Errors (HMM)      - Seg-induced: {hmm_seg_errors}, Genuine: {hmm_gen_errors}, Accuracy: {1 - (hmm_seg_errors+hmm_gen_errors)/max(1, hmm_total):.4f}")
    print(f"POS Tagging Errors (MEMM)     - Seg-induced: {memm_seg_errors}, Genuine: {memm_gen_errors}, Accuracy: {1 - (memm_seg_errors+memm_gen_errors)/max(1, memm_total):.4f}")
    
    # Plot CM for HMM
    os.makedirs("results", exist_ok=True)
    plot_confusion_matrix(all_actual_tags_hmm, all_pred_tags_hmm, hmm.tags, output_file=f"results/{language}_hmm_cm.png")
    
    # Sample test string
    sample = "thequickbrownfoxjumpsoverthelazydog" if is_english else "mispadrespuedenviajar"
    print(f"\nSample Prediction for '{sample}':")
    pred_w = viterbi_seg.segment(sample)
    pred_t = viterbi_hmm.tag(pred_w)
    print(pred_t)

def main():
    print("Loading data...")
    en_train, en_dev, en_test = load_brown_corpus()
    try:
        es_train, es_dev, es_test = get_spanish_data("data")
    except Exception as e:
        print("Spanish data not found. Please ensure it is cloned in data/UD_Spanish-GSD")
        sys.exit(1)
        
    train_and_evaluate("English", en_train, en_test, is_english=True)
    train_and_evaluate("Spanish", es_train, es_test, is_english=False)

if __name__ == "__main__":
    main()
