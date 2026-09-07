import math,time
from dataclasses import dataclass
from .config import Q4Config

@dataclass
class Alert:
    kind:str
    message:str
    latency_ms:float=0.0
    details:dict|None=None

class TriggerGrammarChecker:
    def __init__(self,q3_corrector,shared_lm,config:Q4Config):
        self.q3=q3_corrector;self.lm=shared_lm;self.cfg=config
    def check_window(self,words):
        t0=time.perf_counter();alerts=[]
        clean=[w.lower() for w in words if w]
        if not clean:return alerts,0.0
        # Local word-sequence plausibility: compare average log probability to a
        # conservative threshold. The threshold is intentionally documented and
        # based on the model's smoothed scale rather than a binary grammar rule.
        bg=self.lm.sentence_bigram_logprob(clean);tg=self.lm.sentence_trigram_logprob(clean)
        avg_bg=bg/max(1,len(clean)+1);avg_tg=tg/max(1,len(clean)+1)
        if avg_tg < -11.5 and avg_bg < -9.0:
            alerts.append(Alert('GRAMMAR-ALERT',f'Implausible local word sequence (avg bigram={avg_bg:.3f}, avg trigram={avg_tg:.3f}).',details={'bigram_logprob':bg,'trigram_logprob':tg}))
        # Q3-style real-word correction over the same trigger window.
        real_changes=[]
        working=list(clean)
        for i,w in enumerate(list(working)):
            if w in self.q3.vocab:
                corrected,changed=self.q3.correct_realword(working,i)
                if changed:
                    real_changes.append((w,corrected,i));working[i]=corrected
        if real_changes:
            alerts.append(Alert('GRAMMAR-ALERT','Real-word context candidates detected: '+', '.join(f'{a}->{b}' for a,b,_ in real_changes),details={'real_word_changes':real_changes}))
        return alerts,(time.perf_counter()-t0)*1000
