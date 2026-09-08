from collections import Counter, defaultdict

class GreedyLongestMatchSegmenter:
    def __init__(self, vocab):
        self.vocab = set(vocab)
        self.max_word_len = max([len(w) for w in self.vocab]) if self.vocab else 20
        
    def segment(self, text):
        """
        Segments text by greedily finding the longest matching word from the vocabulary at each step.
        """
        n = len(text)
        words = []
        i = 0
        while i < n:
            match_found = False
            # Try from max possible length down to 1
            max_len = min(self.max_word_len, n - i)
            for j in range(max_len, 0, -1):
                candidate = text[i:i+j]
                if candidate in self.vocab or j == 1:
                    words.append(candidate)
                    i += j
                    match_found = True
                    break
            
            if not match_found:
                # Should not reach here if j=1 is always accepted
                words.append(text[i])
                i += 1
                
        return words

class MostFrequentTagTagger:
    def __init__(self):
        self.word_tag_counts = defaultdict(Counter)
        self.most_frequent_overall = None
        
    def train(self, sentences_with_tags):
        tag_counts = Counter()
        for sentence in sentences_with_tags:
            for word, tag in sentence:
                self.word_tag_counts[word][tag] += 1
                tag_counts[tag] += 1
                
        if tag_counts:
            self.most_frequent_overall = tag_counts.most_common(1)[0][0]
            
    def tag(self, words):
        """
        Tags each word with its most frequent tag seen in training.
        If word is unknown, uses the most frequent overall tag.
        """
        tagged = []
        for word in words:
            if word in self.word_tag_counts:
                best_tag = self.word_tag_counts[word].most_common(1)[0][0]
            else:
                best_tag = self.most_frequent_overall
            tagged.append((word, best_tag))
        return tagged
