import pickle
from collections import defaultdict, Counter
import math
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer

class BaseModel:
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

class TrigramLanguageModel(BaseModel):
    def __init__(self, k=1.0):
        self.k = k # Add-k smoothing parameter
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.vocab = set()
        self.vocab_size = 0
        
    def train(self, sentences):
        """
        sentences: list of lists of words
        """
        for sentence in sentences:
            # We use <s> and </s> for padding
            padded = ['<s>', '<s>'] + sentence + ['</s>']
            for i in range(len(padded)):
                self.vocab.add(padded[i])
                self.unigrams[padded[i]] += 1
                if i >= 1:
                    self.bigrams[(padded[i-1], padded[i])] += 1
                if i >= 2:
                    self.trigrams[(padded[i-2], padded[i-1], padded[i])] += 1
                    
        self.vocab_size = len(self.vocab)

    def score(self, w1, w2, w3):
        """
        Returns the smoothed log probability log P(w3 | w1, w2)
        """
        bigram_count = self.bigrams.get((w1, w2), 0)
        trigram_count = self.trigrams.get((w1, w2, w3), 0)
        
        # Add-k smoothing
        # P(w3 | w1, w2) = (C(w1, w2, w3) + k) / (C(w1, w2) + k * V)
        prob = (trigram_count + self.k) / (bigram_count + self.k * self.vocab_size)
        return math.log(prob)
        
    def get_word_prob(self, word):
        """
        Unigram probability for fallback/unknown words
        """
        count = self.unigrams.get(word, 0)
        prob = (count + self.k) / (sum(self.unigrams.values()) + self.k * self.vocab_size)
        return math.log(prob)

class POSTaggerHMM(BaseModel):
    def __init__(self, k=0.1):
        self.k = k
        self.tags = set()
        self.vocab = set()
        
        # Counts
        self.tag_unigrams = Counter()
        self.tag_bigrams = Counter()
        self.tag_trigrams = Counter()
        self.word_tag_counts = defaultdict(Counter)
        
    def train(self, sentences_with_tags):
        """
        sentences_with_tags: list of lists of (word, tag)
        """
        for sentence in sentences_with_tags:
            # Pad tags for trigram transition
            padded_tags = ['<s>', '<s>'] + [tag for word, tag in sentence] + ['</s>']
            
            for i in range(len(padded_tags)):
                self.tags.add(padded_tags[i])
                self.tag_unigrams[padded_tags[i]] += 1
                if i >= 1:
                    self.tag_bigrams[(padded_tags[i-1], padded_tags[i])] += 1
                if i >= 2:
                    self.tag_trigrams[(padded_tags[i-2], padded_tags[i-1], padded_tags[i])] += 1
                    
            for word, tag in sentence:
                self.vocab.add(word)
                self.word_tag_counts[tag][word] += 1
                
    def transition_prob(self, t1, t2, t3):
        """
        log P(t3 | t1, t2)
        """
        bigram_count = self.tag_bigrams.get((t1, t2), 0)
        trigram_count = self.tag_trigrams.get((t1, t2, t3), 0)
        
        prob = (trigram_count + self.k) / (bigram_count + self.k * len(self.tags))
        return math.log(prob)
        
    def emission_prob(self, word, tag):
        """
        log P(word | tag)
        """
        tag_count = self.tag_unigrams.get(tag, 0)
        word_given_tag_count = self.word_tag_counts[tag].get(word, 0)
        
        prob = (word_given_tag_count + self.k) / (tag_count + self.k * (len(self.vocab) + 1)) # +1 for UNK
        return math.log(prob)

class FeatureBasedPOSTagger(BaseModel):
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=True)
        # Using SGDClassifier for fast training on large sparse datasets (equivalent to LogisticRegression)
        self.classifier = SGDClassifier(loss='log_loss', max_iter=100, n_jobs=-1, random_state=42)
        self.classes_ = None
        self.tags = set()
        
    def extract_features(self, sentence_words, i, prev_tag, prev2_tag):
        word = sentence_words[i]
        features = {
            'word': word,
            'is_first': i == 0,
            'is_last': i == len(sentence_words) - 1,
            'is_capitalized': word[0].isupper() if len(word) > 0 else False,
            'is_all_caps': word.isupper(),
            'is_all_lower': word.islower(),
            'prefix-1': word[0] if len(word) > 0 else '',
            'prefix-2': word[:2] if len(word) > 1 else '',
            'prefix-3': word[:3] if len(word) > 2 else '',
            'suffix-1': word[-1] if len(word) > 0 else '',
            'suffix-2': word[-2:] if len(word) > 1 else '',
            'suffix-3': word[-3:] if len(word) > 2 else '',
            'prev_tag': prev_tag,
            'prev2_tag': prev2_tag,
            'prev_word': sentence_words[i-1] if i > 0 else '<s>',
            'next_word': sentence_words[i+1] if i < len(sentence_words) - 1 else '</s>',
            'has_hyphen': '-' in word,
            'is_numeric': word.isdigit(),
            'capitals_inside': word[1:].lower() != word[1:] if len(word) > 1 else False
        }
        return features

    def train(self, sentences_with_tags):
        X_dicts = []
        Y = []
        
        for sentence in sentences_with_tags:
            words = [word for word, tag in sentence]
            tags = [tag for word, tag in sentence]
            
            for i in range(len(words)):
                prev_tag = tags[i-1] if i > 0 else '<s>'
                prev2_tag = tags[i-2] if i > 1 else '<s>'
                
                features = self.extract_features(words, i, prev_tag, prev2_tag)
                X_dicts.append(features)
                Y.append(tags[i])
                self.tags.add(tags[i])
                
        print("Vectorizing features...")
        X = self.vectorizer.fit_transform(X_dicts)
        X.indices = X.indices.astype(np.int32)
        X.indptr = X.indptr.astype(np.int32)
        print(f"Training Logistic Regression on {X.shape[0]} samples with {X.shape[1]} features...")
        self.classifier.fit(X, Y)
        self.classes_ = self.classifier.classes_
        print("Training complete.")

    def predict_log_proba(self, sentence_words, i, prev_tag, prev2_tag):
        """
        Returns a dictionary of {tag: log_probability}
        """
        features = self.extract_features(sentence_words, i, prev_tag, prev2_tag)
        X = self.vectorizer.transform([features])
        X.indices = X.indices.astype(np.int32)
        X.indptr = X.indptr.astype(np.int32)
        probas = self.classifier.predict_proba(X)[0]
        
        # Convert to log probas safely
        log_probas = {}
        for idx, tag in enumerate(self.classes_):
            # smooth slightly to avoid log(0)
            prob = max(probas[idx], 1e-10)
            log_probas[tag] = math.log(prob)
            
        return log_probas
