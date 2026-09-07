import random
import nltk

def ensure_corpora():
    for name,path in [('gutenberg','corpora/gutenberg'),('brown','corpora/brown'),('reuters','corpora/reuters')]:
        try:nltk.data.find(path)
        except LookupError:
            try:nltk.download(name,quiet=True)
            except Exception:pass

def sample_passage(seed=None,min_sentences=5,max_sentences=8):
    ensure_corpora();rng=random.Random(seed)
    corpora=[]
    try:
        from nltk.corpus import gutenberg
        corpora += [list(s) for s in gutenberg.sents()]
    except Exception:pass
    if len(corpora)<100:
        try:
            from nltk.corpus import brown
            corpora += [list(s) for s in brown.sents()]
        except Exception:pass
    if len(corpora)<100:
        try:
            from nltk.corpus import reuters
            corpora += [list(s) for s in reuters.sents()]
        except Exception:pass
    if not corpora:raise RuntimeError('No NLTK passage corpus is available.')
    n=rng.randint(min_sentences,max_sentences);start=rng.randrange(max(1,len(corpora)-n));return corpora[start:start+n]
