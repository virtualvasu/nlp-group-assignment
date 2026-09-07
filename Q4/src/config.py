from dataclasses import dataclass

@dataclass(frozen=True)
class Q4Config:
    merge_probability: float = 0.08
    grammar_trigger_words: int = 10
    q1_max_word_len: int = 20
    q1_beam_width: int = 5
    q1_alpha: float = 1.0
    q1_beta: float = 1.0
    q4_add_k: float = 0.1
    q3_real_word_margin: float = 1.5
    sleep_seconds: float = 0.20
    random_seed: int | None = None
    pcfg_outlier_z: float = 2.5
