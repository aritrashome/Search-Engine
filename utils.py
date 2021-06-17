import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import hashlib
import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

##################### TOKENIZE ##########################
def tokenize(doc):
    doc = doc.lower()
    tokens = [ps.stem(w) for w in nltk.RegexpTokenizer(r"\w+").tokenize(doc) if not w.lower() in stop_words]
    return tokens
##################### SEGMENTS ##########################
def make_segment_name(term , length=2):
    hashed = hashlib.md5(term.encode('utf-8')).hexdigest()
    return "{0}.json".format(hashed[:length])
###################### Index Reader #########################
def load_segment(term):
    seg_name = make_segment_name(term)
    if not os.path.exists('segments/'+seg_name):
        return {}
    with open('segments/'+seg_name, 'r', encoding="utf-8") as seg_file: 
        for line in seg_file:
            seg_term, term_info = line.rstrip().split('\t',1)
            if seg_term==term:
                return json.loads(term_info)
    return {}

def collect_results(terms):
    per_term_docs = {}
    per_doc_counts = {}
    for term in terms:
        term_matches = load_segment(term)
        per_term_docs.setdefault(term, 0)
        per_term_docs[term] += len(term_matches.keys())
        for doc_id, positions in term_matches.items():
            per_doc_counts.setdefault(doc_id, {})
            per_doc_counts[doc_id].setdefault(term, 0)
            per_doc_counts[doc_id][term] += len(positions)
    return per_term_docs, per_doc_counts
########################### BM25 #############################
def bm25_relevance(terms, matches, current_doc, total_docs, b=0, k=1.2):
    score = b
    for term in terms:
        idf = math.log((total_docs - matches[term] + 1.0) / matches[term]) / math.log(1.0 + total_docs)
        score = score + current_doc.get(term, 0) * idf / (current_doc.get(term, 0) + k)
    return 0.5 + score / (1+2 * len(terms))

 ########################
 #SEARCH
 ########################
def search_paper(query):
 	query_terms  = tokenize(query)
 	per_term_docs, per_doc_counts = collect_results(query_terms)
 	scored_results = []
 	for doc_id, current_doc in per_doc_counts.items():
 		scored_results.append({
 			'id': doc_id,
 			'score': bm25_relevance(query_terms, per_term_docs, current_doc, total_docs=41000),
 		})
 	scored_results = sorted(scored_results, key=lambda res: res['score'], reverse=True)
 	return scored_results