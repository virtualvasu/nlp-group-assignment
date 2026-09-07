import argparse,sys,time,os
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from src.config import Q4Config
from src.model_loader import load_all
from src.ngram_q4 import train_shared_models
from src.passage_sampler import sample_passage
from src.live_pipeline import LiveEditorPipeline,simulate_merged_stream
from src.pcfg_parser import PCFGParser
from src.final_analysis import FinalAnalyzer
from src.benchmark import run_q4_benchmark
from src.reporting import save_json

def build(seed=None):
    cfg=Q4Config(random_seed=seed)
    models=load_all()
    sentences=models['q3_data']['sentences']
    shared=train_shared_models(models['q1_lm'],sentences,k=cfg.q4_add_k)
    pcfg=PCFGParser().train()
    return models,shared,pcfg,cfg

def demo(seed=None, include_benchmark=True):
    models,shared,pcfg,cfg=build(seed)
    raw=sample_passage(seed)
    pipe=LiveEditorPipeline(models['q1_decoder'],models['q3_corrector'],shared,cfg)
    sentence_token_counts=[];merge_counts={};spell_counts={}
    print('\n=== Q4 LIVE DEMO ===')
    for si,sent in enumerate(raw):
        words=[w.lower() for w in sent if w.isalpha()]
        stream=simulate_merged_stream([words],cfg.merge_probability,seed=None if seed is None else seed+si)
        start_len=len(pipe.tokens);m=s=0
        for tok,was_merge in stream:
            item=pipe.process_token(tok)
            m += int(item.segmentation_merge)
            s += int(item.spelling_correction is not None)
            for a in item.alerts:
                print(f'[{a.kind}] {a.message} ({a.latency_ms:.2f} ms)')
            if cfg.sleep_seconds:
                time.sleep(cfg.sleep_seconds)
        sentence_token_counts.append(len(pipe.tokens)-start_len)
        merge_counts[si]=m;spell_counts[si]=s

    # Build final sentence streams from the FINAL corrected token stream.
    # Trigger-level real-word corrections may update an earlier sentence after
    # its tokens were first emitted, so using the original per-sentence snapshots
    # here would produce stale final-analysis input.
    sentence_outputs=[];cursor=0
    for count in sentence_token_counts:
        sentence_outputs.append(pipe.tokens[cursor:cursor+count])
        cursor += count

    # Attribute trigger-level real-word corrections to the sentence containing
    # the corrected token. Token counts are stable because corrections are
    # one-for-one substitutions.
    cumulative=[]
    total=0
    for count in sentence_token_counts:
        total += count;cumulative.append(total)
    for absolute,old,new in pipe.realword_corrections:
        for si,endpos in enumerate(cumulative):
            if absolute < endpos:
                spell_counts[si]=spell_counts.get(si,0)+1
                break

    print('\n=== FINAL ANALYSIS ===')
    analyzer=FinalAnalyzer(models['q1_decoder'],models['q1_tagger'],shared,pcfg)
    df=analyzer.analyze(sentence_outputs,merge_counts,spell_counts)
    print(df.to_string(index=False))
    os.makedirs(ROOT/'results',exist_ok=True)
    label = seed if seed is not None else 'random'
    df.to_csv(ROOT/'results'/f'final_sentence_analysis_seed_{label}.csv',index=False)
    if include_benchmark:
        bench=run_q4_benchmark(models['q1_decoder'],models['q3_corrector'],shared,cfg)
        print('\n=== SPEED DEMON (Q4, exactly 1000 words) ===');print(pd.Series(bench))
        save_json(bench,str(ROOT/'results'/'q4_speed_demon.json'))
    print('\nLive averages:',pipe.averages())

def benchmark():
    # The Speed Demon needs Q1/Q3 + the shared LM, but not the expensive PCFG trainer.
    models=load_all()
    cfg=Q4Config(random_seed=7)
    shared=train_shared_models(models['q1_lm'],models['q3_data']['sentences'],k=cfg.q4_add_k)
    bench=run_q4_benchmark(models['q1_decoder'],models['q3_corrector'],shared,cfg)
    print(pd.Series(bench));save_json(bench,str(ROOT/'results'/'q4_speed_demon.json'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['demo','benchmark']);ap.add_argument('--seed',type=int,default=None);ap.add_argument('--no-benchmark',action='store_true',help='Run demo without the 1,000-word benchmark');args=ap.parse_args()
    demo(args.seed, include_benchmark=not args.no_benchmark) if args.command=='demo' else benchmark()
