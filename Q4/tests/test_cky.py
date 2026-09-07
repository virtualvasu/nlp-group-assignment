import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pcfg_parser import PCFGParser

def test_pcfg_trains_and_handles_empty():
    import nltk
    try:nltk.data.find('corpora/treebank')
    except LookupError:pytest.skip('Penn Treebank not installed in this execution environment')
    p=PCFGParser();p.train();r=p.parse([],[]);assert not r.success
