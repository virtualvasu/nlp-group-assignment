# Q1 uses the Brown Universal POS tagset. Penn Treebank lexical productions
# use PTB tags. This small deterministic mapping is documented for Q4.
Q1_TO_PTB={
 'NOUN':'NN','VERB':'VB','AUX':'VB','ADJ':'JJ','ADV':'RB','DET':'DT','PRON':'PRP',
 'ADP':'IN','NUM':'CD','CONJ':'CC','CCONJ':'CC','SCONJ':'IN','PART':'RP','PRT':'RP','INTJ':'UH',
 'PROPN':'NNP','PUNCT':'.','SYM':'SYM','X':'FW','UNKNOWN':'NN'
}

def q1_to_ptb(tag):
    if tag in Q1_TO_PTB:return Q1_TO_PTB[tag]
    # Brown universal tags occasionally appear as PTB-like tags in legacy artifacts.
    if tag in {'NN','NNS','NNP','NNPS','JJ','JJR','JJS','RB','RBR','RBS','VB','VBD','VBG','VBN','VBP','VBZ','DT','IN','PRP','PRP$','CC','CD','MD','RP','UH','WDT','WP','WP$','WRB','EX','POS','TO'}:
        return tag
    return 'NN'

def map_tagged_tokens(tagged):
    return [(w,q1_to_ptb(t)) for w,t in tagged]
