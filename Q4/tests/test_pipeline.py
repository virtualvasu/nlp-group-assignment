import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.live_pipeline import simulate_merged_stream

def test_merge_stream():
    x=simulate_merged_stream([['the','quick','fox']],1.0,1);assert len(x)==2;assert x[0][0]=='thequick'


class FakeQ1:
    def decode(self, text):
        return [(text, 'NOUN')]


class FakeQ3:
    vocab = {'the', 'cat', 'power'}
    def correct_nonword(self, word):
        return ('power', True) if word == 'pwoer' else (word, False)
    def correct_realword(self, sentence, index):
        return sentence[index], False


def test_oov_spelling_alert_and_real_latency():
    from src.config import Q4Config
    from src.live_pipeline import LiveEditorPipeline
    pipe = LiveEditorPipeline(FakeQ1(), FakeQ3(), object(), Q4Config(grammar_trigger_words=100))
    item = pipe.process_token('pwoer')
    assert item.final_tokens == ['power']
    assert any(a.kind == 'SPELL-ALERT' for a in item.alerts)
    assert item.token_latency_ms >= 0.0
    assert all(a.latency_ms == item.token_latency_ms for a in item.alerts)
