import os
import random
import nltk
from conllu import parse_incr

def load_brown_corpus(split_ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Loads the Brown corpus and splits it into train, dev, and test sets.
    Returns lists of sentences, where each sentence is a list of (word, pos) tuples.
    """
    try:
        sentences = nltk.corpus.brown.tagged_sents(tagset='universal')
    except LookupError:
        nltk.download('brown')
        nltk.download('universal_tagset')
        sentences = nltk.corpus.brown.tagged_sents(tagset='universal')
        
    sentences = list(sentences)
    random.seed(seed)
    random.shuffle(sentences)
    
    n = len(sentences)
    train_end = int(n * split_ratios[0])
    dev_end = train_end + int(n * split_ratios[1])
    
    train_sents = sentences[:train_end]
    dev_sents = sentences[train_end:dev_end]
    test_sents = sentences[dev_end:]
    
    return train_sents, dev_sents, test_sents

def load_conllu_corpus(filepath):
    """
    Loads a CoNLL-U format file and extracts sentences.
    Each sentence is a list of (word, morphology-aware-tag) tuples.
    Tag format: UPOS-Gender-Number (e.g., NOUN-Fem-Sg)
    """
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for tokenlist in parse_incr(f):
            sentence = []
            for token in tokenlist:
                # Skip multiword tokens (which have tuple IDs in conllu)
                if not isinstance(token['id'], int):
                    continue
                    
                word = token['form']
                upos = token['upos']
                feats = token['feats']
                
                tag_parts = [upos]
                if feats:
                    if 'Gender' in feats:
                        tag_parts.append(feats['Gender'])
                    if 'Number' in feats:
                        tag_parts.append(feats['Number'])
                
                tag = "-".join(tag_parts)
                sentence.append((word, tag))
            sentences.append(sentence)
            
    return sentences

def get_spanish_data(data_dir):
    """
    Convenience function to load the Spanish GSD dataset.
    Assumes data_dir contains the cloned UD_Spanish-GSD repo.
    """
    train_path = os.path.join(data_dir, "UD_Spanish-GSD", "es_gsd-ud-train.conllu")
    dev_path = os.path.join(data_dir, "UD_Spanish-GSD", "es_gsd-ud-dev.conllu")
    test_path = os.path.join(data_dir, "UD_Spanish-GSD", "es_gsd-ud-test.conllu")
    
    train_sents = load_conllu_corpus(train_path)
    dev_sents = load_conllu_corpus(dev_path)
    test_sents = load_conllu_corpus(test_path)
    
    return train_sents, dev_sents, test_sents

if __name__ == "__main__":
    # Test loading
    print("Loading Brown corpus...")
    train, dev, test = load_brown_corpus()
    print(f"Brown: {len(train)} train, {len(dev)} dev, {len(test)} test")
    print("Example:", train[0][:3])
    
    try:
        print("\nLoading Spanish corpus...")
        es_train, es_dev, es_test = get_spanish_data("data")
        print(f"Spanish: {len(es_train)} train, {len(es_dev)} dev, {len(es_test)} test")
        print("Example:", es_train[0][:3])
    except Exception as e:
        print("Could not load Spanish corpus:", e)
