import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.ngram_q4 import SmoothedBigramModel

def test_ngram_smoke():
    lm=SmoothedBigramModel(.1);lm.train([['the','cat','sat']]);assert lm.sentence_logprob(['the','cat'])<0
