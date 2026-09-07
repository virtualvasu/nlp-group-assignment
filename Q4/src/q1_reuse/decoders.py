import math

class ViterbiSegmenter:
    def __init__(self,lm,max_word_len=20): self.lm=lm; self.max_word_len=max_word_len
    def segment(self,text):
        n=len(text); dp=[{} for _ in range(n+1)]; dp[0][('<s>','<s>')]=(0.0,[])
        for i in range(n):
            if not dp[i]: continue
            if len(dp[i])>50: dp[i]=dict(sorted(dp[i].items(),key=lambda x:x[1][0],reverse=True)[:50])
            for j in range(i+1,min(i+1+self.max_word_len,n+1)):
                word=text[i:j]
                if word not in self.lm.vocab and len(word)>1: continue
                base=self.lm.get_word_prob(word) if word not in self.lm.vocab else 0
                for (w1,w2),(score,words) in dp[i].items():
                    tp=self.lm.score(w1,w2,word); ns=score+tp+(base-10.0 if word not in self.lm.vocab else 0)
                    state=(w2,word)
                    if state not in dp[j] or ns>dp[j][state][0]: dp[j][state]=(ns,words+[word])
        best=float('-inf'); out=[]
        for (w1,w2),(score,words) in dp[n].items():
            total=score+self.lm.score(w1,w2,'</s>')
            if total>best: best=total; out=words
        return out or [text]

class ViterbiPOSTagger:
    def __init__(self,model): self.model=model; self.is_hmm=hasattr(model,'emission_prob')
    def tag(self,words):
        n=len(words)
        if n==0:return []
        dp=[{} for _ in range(n)]
        if self.is_hmm:
            for tag in self.model.tags:
                if tag not in ['<s>','</s>']:
                    dp[0][('<s>',tag)]=(self.model.transition_prob('<s>','<s>',tag)+self.model.emission_prob(words[0],tag),None)
        else:
            for tag,p in self.model.predict_log_proba(words,0,'<s>','<s>').items():
                if tag not in ['<s>','</s>']: dp[0][('<s>',tag)]=(p,None)
        for i in range(1,n):
            if len(dp[i-1])>50: dp[i-1]=dict(sorted(dp[i-1].items(),key=lambda x:x[1][0],reverse=True)[:50])
            if self.is_hmm:
                for (t0,t1),(score,_) in dp[i-1].items():
                    for t2 in self.model.tags:
                        if t2 in ['<s>','</s>']:continue
                        ns=score+self.model.transition_prob(t0,t1,t2)+self.model.emission_prob(words[i],t2); st=(t1,t2)
                        if st not in dp[i] or ns>dp[i][st][0]:dp[i][st]=(ns,t0)
            else:
                active=list(dp[i-1].items()); features=[self.model.extract_features(words,i,t1,t0) for (t0,t1),_ in active]
                X=self.model.vectorizer.transform(features); X.indices=X.indices.astype('int32'); X.indptr=X.indptr.astype('int32'); probs=self.model.classifier.predict_proba(X)
                for idx,((t0,t1),(score,_)) in enumerate(active):
                    for j,tag in enumerate(self.model.classes_):
                        if tag in ['<s>','</s>']:continue
                        ns=score+math.log(max(probs[idx,j],1e-10)); st=(t1,tag)
                        if st not in dp[i] or ns>dp[i][st][0]:dp[i][st]=(ns,t0)
        best=float('-inf'); state=None
        for (t1,t2),(score,_) in dp[-1].items():
            total=score+(self.model.transition_prob(t1,t2,'</s>') if self.is_hmm else 0)
            if total>best:best=total;state=(t1,t2)
        if state is None:return [(w,'NOUN') for w in words]
        tags=[];cur=state
        for i in range(n-1,-1,-1):
            tags.append(cur[1]); bp=dp[i][cur][1]
            if bp is not None:cur=(bp,cur[0])
        tags.reverse();return list(zip(words,tags))

class JointBeamSearchDecoder:
    def __init__(self,lm,tagger,max_word_len=20,beam_width=5,alpha=1.0,beta=1.0):
        self.lm=lm;self.tagger=tagger;self.max_word_len=max_word_len;self.beam_width=beam_width;self.alpha=alpha;self.beta=beta;self.is_hmm=hasattr(tagger,'emission_prob')
    def decode(self,text):
        n=len(text);beam=[[] for _ in range(n+1)];beam[0].append((0.0,'<s>','<s>','<s>','<s>',[]))
        for i in range(n):
            if not beam[i]:continue
            beam[i].sort(key=lambda x:x[0],reverse=True); current=beam[i][:self.beam_width]
            for j in range(i+1,min(i+1+self.max_word_len,n+1)):
                word=text[i:j]
                if word not in self.lm.vocab and len(word)>1:continue
                base=self.lm.get_word_prob(word) if word not in self.lm.vocab else 0
                for score,w1,w2,t1,t2,hist in current:
                    lm_score=self.lm.score(w1,w2,word)+(base-10.0 if word not in self.lm.vocab else 0)
                    if self.is_hmm:
                        for tag in self.tagger.tags:
                            if tag in ['<s>','</s>']:continue
                            ts=self.tagger.transition_prob(t1,t2,tag)+self.tagger.emission_prob(word,tag)
                            beam[j].append((score+self.alpha*lm_score+self.beta*ts,w2,word,t2,tag,hist+[(word,tag)]))
                    else:
                        words_so_far=[w for w,t in hist]+[word]; idx=len(words_so_far)-1
                        for tag,ts in self.tagger.predict_log_proba(words_so_far,idx,t2,t1).items():
                            if tag in ['<s>','</s>']:continue
                            beam[j].append((score+self.alpha*lm_score+self.beta*ts,w2,word,t2,tag,hist+[(word,tag)]))
            # keep every destination bounded after expanding from i
            for k in range(i+1,min(i+1+self.max_word_len,n+1)):
                if len(beam[k])>self.beam_width*5:
                    beam[k]=sorted(beam[k],key=lambda x:x[0],reverse=True)[:self.beam_width]
        beam[n].sort(key=lambda x:x[0],reverse=True);best=float('-inf');out=[]
        for score,w1,w2,t1,t2,hist in beam[n][:self.beam_width]:
            total=score+self.alpha*self.lm.score(w1,w2,'</s>')+(self.beta*self.tagger.transition_prob(t1,t2,'</s>') if self.is_hmm else 0)
            if total>best:best=total;out=hist
        return out or [(text,next(iter(self.tagger.tags-{ '<s>','</s>' })))]
