import math
import heapq

class ViterbiSegmenter:
    def __init__(self, lm, max_word_len=20):
        self.lm = lm
        self.max_word_len = max_word_len
        
    def segment(self, text):
        """
        Segments a string of characters into words using Viterbi decoding with a Trigram Language Model.
        """
        n = len(text)
        # dp[i] will store a list of best (score, w1, w2, list_of_words) ending at index i
        # Since it's a trigram model, state is defined by the previous two words (w1, w2).
        # We need to maintain the best score for each state (w1, w2) at each position i.
        
        # dp[i][(w1, w2)] = (score, list_of_words_ending_at_w2)
        dp = [{} for _ in range(n + 1)]
        
        # Base case
        dp[0][('<s>', '<s>')] = (0.0, [])
        
        for i in range(n):
            if not dp[i]:
                continue
                
            # Prune search space to top 50 states to prevent exponential blowup
            if len(dp[i]) > 50:
                best_states = sorted(dp[i].items(), key=lambda x: x[1][0], reverse=True)[:50]
                dp[i] = dict(best_states)
                
            for j in range(i + 1, min(i + 1 + self.max_word_len, n + 1)):
                word = text[i:j]
                
                # Heuristic: if word not in vocab, penalize it, unless it's a single character
                if word not in self.lm.vocab and len(word) > 1:
                    continue # Skip long unknown words to speed up, or assign very low prob
                    
                word_prob_base = self.lm.get_word_prob(word) if word not in self.lm.vocab else 0
                
                for (w1, w2), (score, words) in dp[i].items():
                    # Calculate transition log prob
                    transition_prob = self.lm.score(w1, w2, word)
                    
                    # If word is unknown, add a penalty based on its unigram probability
                    if word not in self.lm.vocab:
                        new_score = score + transition_prob + word_prob_base - 10.0
                    else:
                        new_score = score + transition_prob
                        
                    new_state = (w2, word)
                    
                    if new_state not in dp[j] or new_score > dp[j][new_state][0]:
                        dp[j][new_state] = (new_score, words + [word])
                        
        # At the end, find the best path that transitions to </s>
        best_score = float('-inf')
        best_segmentation = []
        
        for (w1, w2), (score, words) in dp[n].items():
            final_prob = self.lm.score(w1, w2, '</s>')
            total_score = score + final_prob
            if total_score > best_score:
                best_score = total_score
                best_segmentation = words
                
        if not best_segmentation:
            return [text]
            
        return best_segmentation

class ViterbiPOSTagger:
    def __init__(self, model):
        self.model = model # Can be POSTaggerHMM or FeatureBasedPOSTagger
        self.is_hmm = hasattr(model, 'emission_prob')
        
    def tag(self, words):
        """
        Tags a list of words using Viterbi decoding.
        """
        import numpy as np
        n = len(words)
        if n == 0:
            return []
            
        dp = [{} for _ in range(n)]
        
        word = words[0]
        if self.is_hmm:
            for tag in self.model.tags:
                if tag in ['<s>', '</s>']: continue
                trans_p = self.model.transition_prob('<s>', '<s>', tag)
                emiss_p = self.model.emission_prob(word, tag)
                dp[0][('<s>', tag)] = (trans_p + emiss_p, None)
        else:
            log_probas = self.model.predict_log_proba(words, 0, '<s>', '<s>')
            for tag, prob in log_probas.items():
                if tag in ['<s>', '</s>']: continue
                dp[0][('<s>', tag)] = (prob, None)
                
        for i in range(1, n):
            if len(dp[i-1]) > 50:
                best_states = sorted(dp[i-1].items(), key=lambda x: x[1][0], reverse=True)[:50]
                dp[i-1] = dict(best_states)
                
            word = words[i]
            
            if self.is_hmm:
                for (t0, t1), (score, _) in dp[i-1].items():
                    for t2 in self.model.tags:
                        if t2 in ['<s>', '</s>']: continue
                        trans_p = self.model.transition_prob(t0, t1, t2)
                        emiss_p = self.model.emission_prob(word, t2)
                        new_score = score + trans_p + emiss_p
                        
                        state = (t1, t2)
                        if state not in dp[i] or new_score > dp[i][state][0]:
                            dp[i][state] = (new_score, t0)
            else:
                # Batch prediction for MEMM
                active_states = list(dp[i-1].items())
                features_list = [self.model.extract_features(words, i, t1, t0) for (t0, t1), _ in active_states]
                
                X = self.model.vectorizer.transform(features_list)
                X.indices = X.indices.astype(np.int32)
                X.indptr = X.indptr.astype(np.int32)
                probas_batch = self.model.classifier.predict_proba(X)
                
                for idx, ((t0, t1), (score, _)) in enumerate(active_states):
                    probas = probas_batch[idx]
                    for j, tag in enumerate(self.model.classes_):
                        if tag in ['<s>', '</s>']: continue
                        prob = math.log(max(probas[j], 1e-10))
                        new_score = score + prob
                        
                        state = (t1, tag)
                        if state not in dp[i] or new_score > dp[i][state][0]:
                            dp[i][state] = (new_score, t0)
                            
        best_score = float('-inf')
        best_state = None
        
        for (t1, t2), (score, _) in dp[n-1].items():
            if self.is_hmm:
                final_trans = self.model.transition_prob(t1, t2, '</s>')
                total_score = score + final_trans
            else:
                total_score = score # no explicit end transition for MEMM usually
                
            if total_score > best_score:
                best_score = total_score
                best_state = (t1, t2)
                
        if best_state is None:
            # Fallback
            return [(w, list(self.model.tags)[0]) for w in words]
            
        # Backtrack
        tags = []
        curr_state = best_state
        for i in range(n-1, -1, -1):
            tags.append(curr_state[1])
            bp = dp[i][curr_state][1]
            if bp is not None:
                curr_state = (bp, curr_state[0])
                
        tags.reverse()
        return list(zip(words, tags))

class JointBeamSearchDecoder:
    def __init__(self, lm, tagger, max_word_len=20, beam_width=5, alpha=1.0, beta=1.0):
        self.lm = lm
        self.tagger = tagger
        self.max_word_len = max_word_len
        self.beam_width = beam_width
        self.alpha = alpha # Weight for LM score
        self.beta = beta   # Weight for Tagger score
        self.is_hmm = hasattr(tagger, 'emission_prob')
        
    def decode(self, text):
        """
        Jointly segments and POS tags the text using beam search.
        Beam state: (score, index, w1, w2, t1, t2, list_of_word_tag_pairs)
        """
        n = len(text)
        
        # beam[i] stores top K states ending at index i
        # State: (score, w1, w2, t1, t2, history)
        beam = [[] for _ in range(n + 1)]
        
        # Init
        beam[0].append((0.0, '<s>', '<s>', '<s>', '<s>', []))
        
        for i in range(n):
            if not beam[i]:
                continue
                
            # Keep only top K for expansion
            beam[i].sort(key=lambda x: x[0], reverse=True)
            current_beam = beam[i][:self.beam_width]
            
            for j in range(i + 1, min(i + 1 + self.max_word_len, n + 1)):
                word = text[i:j]
                
                # Filter out unknown words that are too long to prune search space
                if word not in self.lm.vocab and len(word) > 1:
                    continue
                    
                word_prob_base = self.lm.get_word_prob(word) if word not in self.lm.vocab else 0
                
                for score, w1, w2, t1, t2, history in current_beam:
                    lm_score = self.lm.score(w1, w2, word)
                    if word not in self.lm.vocab:
                        lm_score += word_prob_base - 10.0
                        
                    # Now for tagging
                    if self.is_hmm:
                        for tag in self.tagger.tags:
                            if tag in ['<s>', '</s>']: continue
                            trans_p = self.tagger.transition_prob(t1, t2, tag)
                            emiss_p = self.tagger.emission_prob(word, tag)
                            tag_score = trans_p + emiss_p
                            
                            new_score = score + self.alpha * lm_score + self.beta * tag_score
                            new_state = (new_score, w2, word, t2, tag, history + [(word, tag)])
                            beam[j].append(new_state)
                    else:
                        # MEMM
                        words_so_far = [w for w, t in history] + [word]
                        idx = len(words_so_far) - 1
                        log_probas = self.tagger.predict_log_proba(words_so_far, idx, t2, t1)
                        
                        for tag, tag_score in log_probas.items():
                            if tag in ['<s>', '</s>']: continue
                            new_score = score + self.alpha * lm_score + self.beta * tag_score
                            new_state = (new_score, w2, word, t2, tag, history + [(word, tag)])
                            beam[j].append(new_state)
                            
            # Prune beam[j] occasionally to avoid huge memory? 
            # We already prune at the start of loop i, but doing it inside can help.
            if len(beam[j]) > self.beam_width * 5:
                beam[j].sort(key=lambda x: x[0], reverse=True)
                beam[j] = beam[j][:self.beam_width]
                
        # Finalize
        beam[n].sort(key=lambda x: x[0], reverse=True)
        
        # Add end transitions if HMM
        best_score = float('-inf')
        best_history = []
        
        for score, w1, w2, t1, t2, history in beam[n][:self.beam_width]:
            final_lm = self.lm.score(w1, w2, '</s>')
            final_tag = self.tagger.transition_prob(t1, t2, '</s>') if self.is_hmm else 0.0
            
            total_score = score + self.alpha * final_lm + self.beta * final_tag
            if total_score > best_score:
                best_score = total_score
                best_history = history
                
        if not best_history:
            return [(text, list(self.tagger.tags)[0])]
            
        return best_history
