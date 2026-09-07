"""Run the two required sample passages and the standalone Q4 Speed Demon.

This script saves transcripts/results so the final report can cite actual runs.
"""
import contextlib
import io
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from main import demo, benchmark

results=ROOT/'results';results.mkdir(exist_ok=True)
for seed in (42,123):
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        demo(seed, include_benchmark=False)
    text=buf.getvalue()
    print(text)
    (results/f'demo_seed_{seed}.txt').write_text(text, encoding='utf-8')

buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    benchmark()
text=buf.getvalue()
print(text)
(results/'speed_demon.txt').write_text(text, encoding='utf-8')
print('Saved two demo transcripts and standalone benchmark output under results/.')
