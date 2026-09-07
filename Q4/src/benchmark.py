import random,time
from .live_pipeline import LiveEditorPipeline

def make_corrupted_batch(vocab,n=1000,seed=7):
    rng=random.Random(seed);words=list(vocab);batch=[]
    while len(batch)<n:
        w=rng.choice(words)
        if len(w)<3:continue
        i=rng.randrange(len(w));op=rng.choice(['delete','insert','replace','transpose'])
        if op=='delete':c=w[:i]+w[i+1:]
        elif op=='insert':c=w[:i]+rng.choice('abcdefghijklmnopqrstuvwxyz')+w[i:]
        elif op=='replace':c=w[:i]+rng.choice('abcdefghijklmnopqrstuvwxyz')+w[i+1:]
        else:
            if i==len(w)-1:i-=1
            c=w[:i]+w[i+1]+w[i]+w[i+2:]
        if c not in vocab:batch.append(c)
    return batch

def run_q4_benchmark(q1_decoder,q3_corrector,shared_lm,config,n=1000,seed=7):
    batch=make_corrupted_batch(q3_corrector.vocab,n,seed)
    # Full live layer: segmentation check + spelling check, with no grammar trigger
    # to isolate the added per-token cost.
    import dataclasses
    cfg=dataclasses.replace(config,grammar_trigger_words=n+1)
    pipe=LiveEditorPipeline(q1_decoder,q3_corrector,shared_lm,cfg)
    t0=time.perf_counter();pipe.process_tokens(batch);full=time.perf_counter()-t0
    # Grammar-only benchmark on the exact same 1000-token batch, using one trigger
    # call to avoid repeated trigger scheduling overhead.
    t1=time.perf_counter();
    pipe.grammar.check_window(batch)
    grammar=time.perf_counter()-t1
    return {'n_words':n,'full_pipeline_seconds':full,'full_pipeline_avg_ms':full*1000/n,'grammar_only_seconds':grammar,'grammar_only_avg_ms_per_word':grammar*1000/n,'added_segmentation_spelling_seconds':max(0.0,full-grammar),'added_segmentation_spelling_avg_ms':max(0.0,full-grammar)*1000/n}
