import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.final_analysis import FinalAnalyzer
from src.ngram_q4 import train_shared_models


class FakeLM:
    vocab = {'the', 'cat', 'sat'}
    def get_word_prob(self, word): return -5.0
    def score(self, *args): return -1.0


class FakeTagger:
    tags = {'NOUN', 'DET', 'VERB', '<s>', '</s>'}
    def emission_prob(self, word, tag): return -1.0
    def transition_prob(self, *args): return -1.0


class NoWholeSentenceDecode:
    def decode(self, text):
        raise AssertionError('Final analysis must not invoke the joint decoder on a whole sentence.')


class FakePCFG:
    def parse(self, words, tags):
        return SimpleNamespace(success=True, log_probability=-float(len(words)), tree=None, reason='')


def test_final_analysis_preserves_sentence_boundaries():
    lm = train_shared_models(FakeLM(), [['the', 'cat'], ['the', 'cat', 'sat']])
    analyzer = FinalAnalyzer(NoWholeSentenceDecode(), FakeTagger(), lm, FakePCFG())
    df = analyzer.analyze([['the', 'cat'], ['the', 'cat', 'sat']], {0: 1, 1: 0}, {0: 0, 1: 1})
    assert list(df['sentence']) == ['the cat', 'the cat sat']
    assert list(df['segmentation_merges']) == [1, 0]
    assert list(df['spelling_corrections']) == [0, 1]


def test_universal_prt_maps_to_ptb_rp():
    from src.tag_mapping import q1_to_ptb
    assert q1_to_ptb('PRT') == 'RP'
