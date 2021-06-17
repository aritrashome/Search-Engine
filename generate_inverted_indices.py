import numpy as np
import pandas as pd
from tqdm import tqdm
import hashlib
import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

df = pd.read_csv('arxiv_data.csv')

##################### TOKENIZE ##########################
def tokenize(doc):
    doc = doc.lower()
    tokens = [ps.stem(w) for w in nltk.RegexpTokenizer(r"\w+").tokenize(doc) if not w.lower() in stop_words]
    return tokens
##################### SEGMENTS ##########################
def make_segment_name(term , length=1):
    hashed = hashlib.md5(term.encode('utf-8')).hexdigest()
    return "{0}.json".format(hashed[:length])

def save_segment(term, term_info):
    seg_name = make_segment_name(term)
    with open('segments/'+seg_name, 'a', encoding="utf-8") as seg_file: 
        seg_file.write("{0}\t{1}\n".format(term, json.dumps(term_info)))
##################### Creating Inverted Indices ################
terms = {}
for i in tqdm(range(len(df))):
    tokens = tokenize( df['title'][i]+' '+df['summary'][i] )
    doc_id = i
    for position, token in enumerate(tokens):
        terms.setdefault(token, {})
        terms[token].setdefault(doc_id, [])
        terms[token][doc_id].append(position)

##################### Saving Inverted Indices ################
for key in tqdm(terms):
    save_segment(key, terms[key])