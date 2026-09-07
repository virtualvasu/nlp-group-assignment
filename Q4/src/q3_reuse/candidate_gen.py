from collections import defaultdict
LETTERS='abcdefghijklmnopqrstuvwxyz'
def edits1(word):
    splits=[(word[:i],word[i:]) for i in range(len(word)+1)]
    deletes=[L+R[1:] for L,R in splits if R]
    transposes=[L+R[1]+R[0]+R[2:] for L,R in splits if len(R)>1]
    replaces=[L+c+R[1:] for L,R in splits if R for c in LETTERS]
    inserts=[L+c+R for L,R in splits for c in LETTERS]
    return set(deletes+transposes+replaces+inserts)
def method_a_candidates(word,vocab):return {w for w in edits1(word) if w in vocab}
def _deletes_of(word):return {word[:i]+word[i+1:] for i in range(len(word))} if word else set()
def build_symspell_dictionary(vocab):
    d=defaultdict(list)
    for w in vocab:
        d[w].append(w)
        for x in _deletes_of(w):d[x].append(w)
    return d
def _levenshtein_le1(a,b):
    if a==b:return True
    if abs(len(a)-len(b))>1:return False
    if len(a)>len(b):a,b=b,a
    i=j=0;edited=False
    while i<len(a) and j<len(b):
        if a[i]==b[j]:i+=1;j+=1;continue
        if edited:return False
        edited=True
        if len(a)==len(b):i+=1;j+=1
        else:j+=1
    return True
def method_b_candidates(word,delete_dict,vocab):
    found=set()
    for variant in _deletes_of(word)|{word}:
        found.update(delete_dict.get(variant,()))
    return {c for c in found if c!=word and _levenshtein_le1(word,c)}
