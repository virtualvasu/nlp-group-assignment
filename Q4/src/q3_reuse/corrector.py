import re
from .candidate_gen import build_symspell_dictionary,method_a_candidates,method_b_candidates
from .data_prep import phrase_log_prob
WORD_RE=re.compile(r'[a-zA-Z]+');REAL_WORD_LOG_MARGIN=1.5
class SpellingCorrector:
    def __init__(self,vocab,freq,bigram_counts,unigram_counts,vocab_size):
        self.vocab=vocab;self.freq=freq;self.bigram_counts=bigram_counts;self.unigram_counts=unigram_counts;self.vocab_size=vocab_size;self.delete_dict=build_symspell_dictionary(vocab)
    def candidates(self,word):return method_a_candidates(word,self.vocab)|method_b_candidates(word,self.delete_dict,self.vocab)
    def candidates_method_a(self,word):return method_a_candidates(word,self.vocab)
    def candidates_method_b(self,word):return method_b_candidates(word,self.delete_dict,self.vocab)
    def correct_nonword(self,word):
        lw=word.lower()
        if lw in self.vocab:return word,False
        c=self.candidates(lw)
        if not c:return word,False
        return max(c,key=lambda w:self.freq.get(w,0)),True
    def correct_realword(self,sentence_words,index):
        word=sentence_words[index]
        if word not in self.vocab:return word,False
        c=self.candidates(word)
        if not c:return word,False
        def phrase_with(w):
            trial=list(sentence_words);trial[index]=w;return trial[max(0,index-1):min(len(trial),index+2)]
        orig=phrase_log_prob(phrase_with(word),self.bigram_counts,self.unigram_counts,self.vocab_size);best_word,best=word,orig
        for cand in c:
            lp=phrase_log_prob(phrase_with(cand),self.bigram_counts,self.unigram_counts,self.vocab_size)
            if lp>best:best_word,best=cand,lp
        return (best_word,True) if best_word!=word and best-orig>REAL_WORD_LOG_MARGIN else (word,False)
    def correct_sentence(self,sentence):
        spans=list(WORD_RE.finditer(sentence));lowered=[m.group(0).lower() for m in spans];out=list(lowered);changes=[]
        for i,w in enumerate(lowered):
            corrected,changed=self.correct_nonword(w) if w not in self.vocab else self.correct_realword(lowered,i)
            out[i]=corrected if changed else spans[i].group(0)
            if changed:changes.append((spans[i].group(0),corrected))
        result=[];last=0
        for m,w in zip(spans,out):result += [sentence[last:m.start()],w];last=m.end()
        result.append(sentence[last:]);return ''.join(result),changes
