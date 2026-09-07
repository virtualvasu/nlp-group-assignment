import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def compute_segmentation_accuracy(predicted_words, gold_words):
    """
    Computes accuracy as the percentage of words segmented exactly right.
    Uses basic overlap or sequence matching.
    """
    if not gold_words:
        return 0.0
    
    # A simple metric: character-level boundaries
    def get_boundaries(words):
        boundaries = set()
        curr = 0
        for w in words[:-1]:
            curr += len(w)
            boundaries.add(curr)
        return boundaries
        
    pred_b = get_boundaries(predicted_words)
    gold_b = get_boundaries(gold_words)
    
    # F1 score for boundaries
    if not gold_b and not pred_b:
        return 1.0
    if not gold_b or not pred_b:
        return 0.0
        
    correct = len(pred_b.intersection(gold_b))
    precision = correct / len(pred_b) if pred_b else 0
    recall = correct / len(gold_b) if gold_b else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def analyze_errors(predicted_tags, gold_tags):
    """
    Separates tagging errors caused by segmentation from genuine tagging errors.
    predicted_tags: list of (word, tag)
    gold_tags: list of (word, tag)
    Returns: (segmentation_induced_errors, genuine_tagging_errors, total_words)
    """
    # Character alignment to match words
    gold_str = "".join([w for w, t in gold_tags])
    pred_str = "".join([w for w, t in predicted_tags])
    
    if gold_str != pred_str:
        # If texts don't match exactly at char level, it's problematic
        pass
        
    # Build maps of char_start_index -> (word, tag, char_end_index)
    def build_map(tags):
        idx_map = {}
        curr = 0
        for w, t in tags:
            idx_map[curr] = (w, t, curr + len(w))
            curr += len(w)
        return idx_map
        
    gold_map = build_map(gold_tags)
    pred_map = build_map(predicted_tags)
    
    segmentation_errors = 0
    genuine_errors = 0
    total = len(gold_tags)
    
    actual_tags = []
    pred_tags = []
    
    for start_idx, (g_w, g_t, g_end) in gold_map.items():
        if start_idx in pred_map:
            p_w, p_t, p_end = pred_map[start_idx]
            if p_end == g_end:
                # Correct segmentation
                if p_t != g_t:
                    genuine_errors += 1
                actual_tags.append(g_t)
                pred_tags.append(p_t)
            else:
                # Wrong segmentation
                segmentation_errors += 1
        else:
            # Wrong segmentation (start index doesn't even match)
            segmentation_errors += 1
            
    return segmentation_errors, genuine_errors, total, actual_tags, pred_tags

def plot_confusion_matrix(actual, predicted, tags, output_file="confusion_matrix.png"):
    """
    Plots and saves a confusion matrix.
    """
    # Get sorted unique tags present in actual or predicted
    unique_tags = sorted(list(set(actual) | set(predicted)))
    
    cm = confusion_matrix(actual, predicted, labels=unique_tags)
    
    # Adjust formatting based on number of tags
    n_tags = len(unique_tags)
    figsize = (15, 12) if n_tags > 20 else (10, 8)
    annot = False if n_tags > 20 else True
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='d' if annot else '', 
                xticklabels=unique_tags, yticklabels=unique_tags, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('POS Tagging Confusion Matrix (Genuine Errors Only)')
    
    # Rotate x labels for better readability if many tags
    if n_tags > 20:
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)
        
    plt.tight_layout()
    plt.savefig(output_file, dpi=150 if n_tags > 20 else 100)
    plt.close()
    
    print(f"Confusion matrix saved to {output_file}")
