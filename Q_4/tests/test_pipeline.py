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


class TriggerQ3:
    vocab = {'w' + str(i) for i in range(20)}
    def correct_nonword(self, word):
        return word, False
    def correct_realword(self, sentence, index):
        if sentence[index] == 'w9':
            return 'corrected', True
        return sentence[index], False


class TriggerQ1:
    def decode(self, text):
        return [(text, 'NOUN')]


class AlwaysBadLM:
    def sentence_bigram_logprob(self, words):
        return -100.0
    def sentence_trigram_logprob(self, words):
        return -100.0


def test_grammar_triggers_exactly_on_multiples_of_n():
    from src.config import Q4Config
    from src.live_pipeline import LiveEditorPipeline
    cfg = Q4Config(grammar_trigger_words=10)
    pipe = LiveEditorPipeline(TriggerQ1(), TriggerQ3(), AlwaysBadLM(), cfg)

    pipe.process_tokens(['w0'] * 9)
    assert len(pipe.trigger_latencies) == 0

    pipe.process_token('w9')
    assert len(pipe.trigger_latencies) == 1

    pipe.process_tokens(['w0'] * 9)
    assert len(pipe.trigger_latencies) == 1

    pipe.process_token('w0')
    assert len(pipe.trigger_latencies) == 2
    assert pipe.tokens[9] == 'corrected'
    assert pipe.averages()['total_processing_ms'] >= 0.0


def test_real_word_correction_mutates_final_stream():
    from src.config import Q4Config
    from src.live_pipeline import LiveEditorPipeline
    pipe = LiveEditorPipeline(
        TriggerQ1(), TriggerQ3(), AlwaysBadLM(),
        Q4Config(grammar_trigger_words=10),
    )
    pipe.process_tokens(['w0'] * 9)
    pipe.process_token('w9')

    assert pipe.tokens[9] == 'corrected'
    assert pipe.realword_corrections == [(9, 'w9', 'corrected')]
