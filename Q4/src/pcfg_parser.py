import math
from collections import defaultdict,Counter
from dataclasses import dataclass
import nltk
from nltk.tree import Tree
from nltk.grammar import induce_pcfg,Production,Nonterminal

@dataclass
class ParseResult:
    success:bool
    log_probability:float|None
    tree:Tree|None
    reason:str=''

class PCFGParser:
    """PCFG induction + custom CKY/Viterbi parser.

    Training uses Penn Treebank sample trees transformed to Chomsky Normal Form.
    Parsing uses the supplied/reconciled PTB preterminal sequence, binary-rule
    dynamic programming, and unary closure. The highest-probability derivation is
    retained for every chart cell.
    """
    def __init__(self):
        self.grammar=None;self.binary=defaultdict(list);self.unary=defaultdict(list);self.lexical=defaultdict(list)
        self.lexical_by_tag=defaultdict(list);self.start=None;self.rule_counts=Counter();self.trained=False
    def train(self):
        try:nltk.data.find('corpora/treebank')
        except LookupError:nltk.download('treebank',quiet=True)
        from nltk.corpus import treebank
        productions=[];start=Nonterminal('S')
        for original in treebank.parsed_sents():
            t=original.copy(deep=True)
            # Strip top wrapper labels like S -> ... while preserving S as root.
            try:t.chomsky_normal_form(horzMarkov=2,vertMarkov=1)
            except TypeError:t.chomsky_normal_form()
            productions.extend(t.productions())
        self.grammar=induce_pcfg(start,productions);self.start=self.grammar.start();self._index();self.trained=True;return self
    def _index(self):
        self.binary.clear();self.unary.clear();self.lexical.clear();self.lexical_by_tag.clear()
        for p in self.grammar.productions():
            lhs=p.lhs();rhs=p.rhs();
            if len(rhs)==2 and all(isinstance(x,Nonterminal) for x in rhs):self.binary[(rhs[0],rhs[1])].append((lhs,math.log(p.prob())))
            elif len(rhs)==1 and isinstance(rhs[0],Nonterminal):self.unary[rhs[0]].append((lhs,math.log(p.prob())))
            elif len(rhs)==1:
                word=str(rhs[0]).lower();self.lexical[word].append((lhs,math.log(p.prob())));self.lexical_by_tag[lhs].append((word,math.log(p.prob())))
    def _unary_closure(self,cell):
        changed=True
        while changed:
            changed=False
            items=list(cell.items())
            for child,(score,tree) in items:
                for parent,lp in self.unary.get(child,[]):
                    ns=score+lp
                    if parent not in cell or ns>cell[parent][0]:
                        cell[parent]=(ns,Tree(str(parent),[tree]));changed=True
        return cell
    def parse(self,words,ptb_tags):
        if not self.trained:self.train()
        n=len(words)
        if n==0:return ParseResult(False,None,None,'empty sentence')
        chart=[[{} for _ in range(n+1)] for __ in range(n)]
        for i,(word,tag) in enumerate(zip(words,ptb_tags)):
            pos=Nonterminal(tag);w=word.lower()
            lexical_options=[(lhs,lp) for lhs,lp in self.lexical.get(w,[]) if lhs==pos]
            if lexical_options:
                # Multiple identical lexical rules cannot occur after PCFG induction,
                # but retaining max is robust.
                lp=max(x[1] for x in lexical_options);chart[i][i+1][pos]=(lp,Tree(tag,[word]))
            else:
                # Unknown lexical item: use the reconciled POS as a tiny backoff
                # lexical probability. This keeps the parser graceful on Gutenberg
                # words absent from the PTB sample without pretending the word was seen.
                if self.lexical_by_tag.get(pos):
                    avg=sum(math.exp(lp) for _,lp in self.lexical_by_tag[pos])/len(self.lexical_by_tag[pos]);lp=math.log(max(avg*1e-4,1e-12))
                else:
                    # If this PTB tag itself is absent, try NN as a safe lexical backoff.
                    pos=Nonterminal('NN');lp=math.log(1e-12)
                chart[i][i+1][pos]=(lp,Tree(str(pos),[word]))
            self._unary_closure(chart[i][i+1])
        for span in range(2,n+1):
            for i in range(n-span+1):
                j=i+span
                cell=chart[i][j]
                for k in range(i+1,j):
                    left=chart[i][k];right=chart[k][j]
                    for (b,(sb,tb)) in left.items():
                        for (c,(sc,tc)) in right.items():
                            for a,lp in self.binary.get((b,c),[]):
                                ns=sb+sc+lp
                                if a not in cell or ns>cell[a][0]:cell[a]=(ns,Tree(str(a),[tb,tc]))
                self._unary_closure(cell)
        start_key=self.start
        if start_key not in chart[0][n]:
            # Some treebank wrappers may have an alternate root after CNF.
            candidates=[(nt,val) for nt,val in chart[0][n].items() if str(nt) in {'S','TOP'}]
            if not candidates:return ParseResult(False,None,None,'no complete S derivation')
            nt,(score,tree)=max(candidates,key=lambda x:x[1][0])
            return ParseResult(True,score,tree)
        score,tree=chart[0][n][start_key];return ParseResult(True,score,tree)
