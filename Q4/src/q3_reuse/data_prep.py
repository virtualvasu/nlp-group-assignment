import os,pickle,re,math
from collections import Counter
import nltk
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR=os.path.join(BASE,'artifacts','q3_data');os.makedirs(DATA_DIR,exist_ok=True)
VOCAB_PATH=os.path.join(DATA_DIR,'vocab_freq.pkl');BIGRAM_PATH=os.path.join(DATA_DIR,'bigram_model.pkl');SENTENCES_PATH=os.path.join(DATA_DIR,'sentences.pkl')
WORD_RE=re.compile(r'^[a-z]+$')
def _ensure_brown_downloaded():
    try:nltk.data.find('corpora/brown')
    except LookupError:nltk.download('brown')
def load_brown_sentences():
    _ensure_brown_downloaded();from nltk.corpus import brown
    return [[w.lower() for w in sent if WORD_RE.match(w.lower())] for sent in brown.sents() if any(WORD_RE.match(w.lower()) for w in sent)]
def build_vocab_and_freq(sentences):
    freq=Counter();[freq.update(s) for s in sentences];return set(freq.keys()),freq
def build_bigram_model(sentences,vocab):
    bc=Counter();uc=Counter()
    for sent in sentences:
        padded=['<s>']+sent+['</s>']
        for w1,w2 in zip(padded,padded[1:]):bc[(w1,w2)]+=1;uc[w1]+=1
    return bc,uc,len(vocab)+2
def bigram_prob(w1,w2,bigram_counts,unigram_counts,vocab_size):return (bigram_counts.get((w1,w2),0)+1)/(unigram_counts.get(w1,0)+vocab_size)
def phrase_log_prob(words,bigram_counts,unigram_counts,vocab_size):
    p=0.0;padded=['<s>']+words+['</s>']
    for w1,w2 in zip(padded,padded[1:]):p+=math.log(bigram_prob(w1,w2,bigram_counts,unigram_counts,vocab_size))
    return p
def build_and_cache_all(force_rebuild=False):
    if not force_rebuild and all(os.path.exists(p) for p in [VOCAB_PATH,BIGRAM_PATH,SENTENCES_PATH]):
        with open(VOCAB_PATH,'rb') as f:vocab,freq=pickle.load(f)
        with open(BIGRAM_PATH,'rb') as f:bc,uc,vs=pickle.load(f)
        with open(SENTENCES_PATH,'rb') as f:sents=pickle.load(f)
        return {'sentences':sents,'vocab':vocab,'freq':freq,'bigram_counts':bc,'unigram_counts':uc,'vocab_size':vs}
    sents=load_brown_sentences();vocab,freq=build_vocab_and_freq(sents);bc,uc,vs=build_bigram_model(sents,vocab)
    for path,obj in [(VOCAB_PATH,(vocab,freq)),(BIGRAM_PATH,(bc,uc,vs)),(SENTENCES_PATH,sents)]:
        with open(path,'wb') as f:pickle.dump(obj,f)
    return {'sentences':sents,'vocab':vocab,'freq':freq,'bigram_counts':bc,'unigram_counts':uc,'vocab_size':vs}
