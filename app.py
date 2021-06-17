from flask import Flask, render_template, url_for, request, redirect
import time
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
from utils import search_paper
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

df = pd.read_csv('arxiv_data.csv')

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html', )

@app.route('/search', methods=['POST', 'GET'])
def search():
	start_time = time.time()
	query = request.form['search']
	scored_results = search_paper(query)
	results = []
	for i in range(min(10, len(scored_results))):
		id = int(scored_results[i]['id'])
		results.append( [ df['title'][id], df['summary'][id] ] )
	return render_template('search.html', query=query, results=results, time=time.time() - start_time)

if __name__ == "__main__":
	#from waitress import serve
	#serve(app, host="0.0.0.0", port=8080)
	app.run(debug=True)