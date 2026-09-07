import os,pickle,sys
from pathlib import Path
from .q1_reuse.ngram_models import TrigramLanguageModel,FeatureBasedPOSTagger
from .q1_reuse.decoders import JointBeamSearchDecoder
from .q3_reuse.corrector import SpellingCorrector

ROOT=Path(__file__).resolve().parents[1]
Q1_DIR=ROOT/'artifacts'/'q1_models';Q3_DIR=ROOT/'artifacts'/'q3_data'

def load_q1_decoder():
    lm_path=Q1_DIR/'english_trigram_lm.pkl';tag_path=Q1_DIR/'english_memm_tagger.pkl'
    if not lm_path.exists() or not tag_path.exists():
        raise FileNotFoundError('Q1 artifacts missing. Run `python prepare_models.py` or copy the original Q1 English model files into artifacts/q1_models/.')
    # Original Q1 pickle files were created with top-level module names
    # `ngram_models` / `decoders`. Register compatibility aliases before loading.
    from .q1_reuse import ngram_models as _nm
    sys.modules.setdefault('ngram_models', _nm)
    lm=TrigramLanguageModel.load(lm_path);tagger=FeatureBasedPOSTagger.load(tag_path)
    return JointBeamSearchDecoder(lm,tagger,max_word_len=20,beam_width=5,alpha=1.0,beta=1.0),lm,tagger

def load_q3_corrector():
    paths=[Q3_DIR/'vocab_freq.pkl',Q3_DIR/'bigram_model.pkl',Q3_DIR/'sentences.pkl']
    if not all(p.exists() for p in paths):
        raise FileNotFoundError('Q3 artifacts missing. Run `python prepare_models.py` or copy the original Q3 data artifacts into artifacts/q3_data/.')
    with open(paths[0],'rb') as f:vocab,freq=pickle.load(f)
    with open(paths[1],'rb') as f:bc,uc,vs=pickle.load(f)
    with open(paths[2],'rb') as f:sentences=pickle.load(f)
    return SpellingCorrector(vocab,freq,bc,uc,vs),{'sentences':sentences,'vocab':vocab,'freq':freq,'bigram_counts':bc,'unigram_counts':uc,'vocab_size':vs}

def load_all():
    q1,lm,tagger=load_q1_decoder();q3,q3data=load_q3_corrector()
    return {'q1_decoder':q1,'q1_lm':lm,'q1_tagger':tagger,'q3_corrector':q3,'q3_data':q3data}
