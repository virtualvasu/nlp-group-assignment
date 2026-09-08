import math
from collections import Counter

class SmoothedBigramModel:
    def __init__(self,k=0.1):self.k=k;self.unigrams=Counter();self.bigrams=Counter();self.vocab=set();self.V=0
    def train(self,sentences):
        for sent in sentences:
            p=['<s>']+[w.lower() for w in sent if w]+['</s>']
            for i,w in enumerate(p):
                self.unigrams[w]+=1;self.vocab.add(w)
                if i:self.bigrams[(p[i-1],w)]+=1
        self.V=len(self.vocab)
    def log_prob(self,w1,w2):return math.log((self.bigrams.get((w1,w2),0)+self.k)/(self.unigrams.get(w1,0)+self.k*self.V))
    def sentence_logprob(self,words):
        p=['<s>']+[w.lower() for w in words]+['</s>'];return sum(self.log_prob(a,b) for a,b in zip(p,p[1:]))

class SharedNGramLM:
    """Q4 shared LM facade.

    The trigram component is the trained Q1 English TrigramLanguageModel and is
    therefore reused unchanged for segmentation and sentence-level trigram
    analysis. The bigram component is trained once here from Brown (using the
    same cleaned Brown sentence stream used by Q3).
    """
    def __init__(self,q1_trigram,k=0.1):self.trigram=q1_trigram;self.bigram=SmoothedBigramModel(k)
    def train_bigram(self,sentences):self.bigram.train(sentences);return self
    def bigram_log_prob(self,w1,w2):return self.bigram.log_prob(w1,w2)
    def trigram_log_prob(self,w1,w2,w3):return self.trigram.score(w1,w2,w3)
    def sentence_bigram_logprob(self,words):return self.bigram.sentence_logprob(words)
    def sentence_trigram_logprob(self,words):
        p=['<s>','<s>']+[w.lower() for w in words]+['</s>'];return sum(self.trigram.score(a,b,c) for a,b,c in zip(p,p[1:],p[2:]))
    def perplexity(self,logp,n):return math.exp(-logp/max(1,n))

def train_shared_models(q1_trigram,sentences,k=0.1):return SharedNGramLM(q1_trigram,k).train_bigram(sentences)
