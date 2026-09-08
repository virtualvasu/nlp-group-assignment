import math, statistics
from dataclasses import dataclass
import pandas as pd
from .tag_mapping import map_tagged_tokens
from .q1_reuse.decoders import ViterbiPOSTagger

@dataclass
class SentenceAnalysis:
    sentence: str
    pcfg_result: str
    pcfg_logprob: float | None
    bigram_logprob: float
    trigram_logprob: float
    chosen_method: str
    verdict: str
    segmentation_merges: int
    spelling_corrections: int

class FinalAnalyzer:
    """Sentence-level Q4 analysis over the already-finalized token stream.

    Important integration detail: Q1's JointBeamSearchDecoder is used for
    *merged-token repair during live processing*. It must not be run over an
    entire normal sentence with spaces removed. At final analysis time we keep
    the corrected token boundaries and only use Q1's feature-based POS model
    to obtain tags for PCFG tagset reconciliation.
    """
    def __init__(self, q1_decoder, q1_tagger, shared_lm, pcfg_parser):
        self.q1 = q1_decoder
        self.tagger = q1_tagger
        self.lm = shared_lm
        self.pcfg = pcfg_parser
        self.pos_tagger = ViterbiPOSTagger(q1_tagger)

    def analyze(self, sentences, merge_counts=None, spell_counts=None):
        merge_counts = merge_counts or {}
        spell_counts = spell_counts or {}
        intermediate = []
        pcfg_scores = []

        for idx, sentence in enumerate(sentences):
            words = [str(w).lower() for w in sentence if str(w).strip()]
            if not words:
                continue

            # Preserve final token boundaries. POS-tag those tokens directly
            # using Q1's trained feature-based tagger; do NOT concatenate the
            # sentence and invoke the joint segmentation decoder.
            tagged = self.pos_tagger.tag(words)
            ptb_tagged = map_tagged_tokens(tagged)
            ptb_tags = [tag for _, tag in ptb_tagged]

            result = self.pcfg.parse(words, ptb_tags)
            bg = self.lm.sentence_bigram_logprob(words)
            tg = self.lm.sentence_trigram_logprob(words)
            intermediate.append((idx, words, result, bg, tg))

            if result.success:
                pcfg_scores.append(result.log_probability / max(1, len(words)))

        threshold = None
        if len(pcfg_scores) >= 3:
            mu = statistics.mean(pcfg_scores)
            sd = statistics.pstdev(pcfg_scores)
            threshold = mu - 2.5 * (sd if sd > 0 else 1e-9)

        rows = []
        for idx, words, result, bg, tg in intermediate:
            norm = result.log_probability / max(1, len(words)) if result.success else None
            if result.success and (threshold is None or norm >= threshold):
                method = 'PCFG'
                score = norm
            elif math.isfinite(tg):
                method = 'TRIGRAM'
                score = tg / max(1, len(words))
            else:
                method = 'BIGRAM'
                score = bg / max(1, len(words))

            # This is intentionally a model-based grammaticality verdict, not
            # a claim of linguistic truth. PCFG success is treated as the
            # strongest signal; n-gram fallbacks use a documented threshold.
            if method == 'PCFG':
                verdict = 'LIKELY-GRAMMATICAL'
            else:
                verdict = 'POTENTIALLY-UNGRAMMATICAL' if score < -10.0 else 'LIKELY-GRAMMATICAL'

            rows.append(SentenceAnalysis(
                sentence=' '.join(words),
                pcfg_result='PARSED' if result.success else 'UNPARSEABLE',
                pcfg_logprob=result.log_probability,
                bigram_logprob=bg,
                trigram_logprob=tg,
                chosen_method=method,
                verdict=verdict,
                segmentation_merges=merge_counts.get(idx, 0),
                spelling_corrections=spell_counts.get(idx, 0),
            ).__dict__)

        return pd.DataFrame(rows)
