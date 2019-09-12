#!/usr/bin/env python
# coding: utf-8

# In[109]:


import math
import os
import time
import re
import sys
import string
import operator
import heapq
from collections import Counter,defaultdict
from datetime import datetime
from nltk.stem.porter import PorterStemmer
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import heappush, heappop
from numpy.linalg import norm

nlp = English()
nlp.max_length = 1000000000
porter_stemmer = PorterStemmer()
query_re = re.compile(r'[t|b|c|e|i|r]:')
vsm_option={"pivot_len_normalization":True,"cosine_normalization":False}
field_map={"title:":"t:","body:":"b:","category:":"c:","ref:":"r:","infobox:":"i:","link:":"e:"}
DOC_CTR = open("docid_ctr","r").read().split("\n")[0].strip()
DOC_CTR=int(DOC_CTR)
token_stats={}
## title should be weighted more or equal than body and then comes the references which is weighted more than infobox which is more than category and external links
field_weights1={'t': 0.3,'b': 0.25,'i': 0.15,'c': 0.05,'e': 0.05,'r': 0.2} ## title > body > ref > infobox > cat = extlink
field_weights2={'t': 1/6,'b': 1/6,'i': 1/6,'c': 1/6,'e': 1/6,'r': 1/6} ## all are equally important
field_weights3={'t': 0.3,'b': 0.25,'i': 0.10,'c': 0.05,'e': 0.10,'r': 0.2} ## title > body > ref > infobox = extlink > cat
field_weights4={'t': 0.3,'b': 0.3,'i': 0.10,'c': 0.05,'e': 0.05,'r': 0.2} ## title = body > ref > infobox > cat = extlink
field_weights5={'t': 1,'b': 1,'i': 1,'c': 1,'e': 1,'r': 1} ## do not apply zone weighting
field_weights=field_weights1
def read_token_stats(path_to_index):
	for file in os.listdir(path_to_index):
		if "docid_token_stats_map-" in file:
			with open(os.path.join(path_to_index,file),"r") as f:
				for line in f:
					line=line.strip()
					if line:
						line = line.split(";")
						token_stats[line[0]]=line[1]
			f.close()

def get_time_info(sec_elapsed):
	h = int(sec_elapsed / (60 * 60))
	m = int((sec_elapsed % (60 * 60)) / 60)
	s = sec_elapsed % 60
	return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def tf_log(token_freq):
	return 1 + math.log10(token_freq)

def compute_idf(token_num_docs):
	return math.log10(DOC_CTR/token_num_docs)

def max_tf_normalization(token_freq,max_term_freq,log_enable=True):
	if log_enable:
		return 0.4 + 0.6*(tf_log(token_freq)/tf_log(max_term_freq))
	return 0.4 + 0.6*(token_freq/max_term_freq)

def avg_tf_normalization(token_freq,avg_token_freq):
	return tf_log(token_freq)/tf_log(avg_token_freq)

def cosine_norm_factor(vec):
	return norm(list(vec.values()))

def vsm(tf_idf,options,norms):
	if options["pivot_len_normalization"]: ## norm_factor = (1-m)avg(norm) + m*(norm), norm is document normalization term
		m=0.75 ##slope
		avg_norm = sum(norms.values())/len(norms.keys())
	elif options["cosine_normalization"]:
		cosine_norm_factor = cosine_norm_factor(tf_idf)
	for docid in tf_idf:
		norm_factor = norms[docid]
		if options["pivot_len_normalization"]:
			norm_factor*=m
			norm_factor+=(1-m)*avg_norm
		elif options["cosine_normalization"]:
			norm_factor = cosine_norm_factor
		tf_idf[docid]/=norm_factor
	return tf_idf


def calc_tfidf(documents,vsm_options=None):
	tf_idf = defaultdict(float) ## it stores tfidf score of retrieved documents for a query
	norms=defaultdict(float)
	for term in documents: ## 'term' here means index_word
		num_term_docs = len(documents[term]) ## 'num_term_docs' is number of documents in which word present
		if num_term_docs: ## if num_term_docs is atleast 1, because term's absence from any document will not contribute to search
			if vsm_options:
				idf=1
			else:
				idf = compute_idf(num_term_docs) ## DOC_CTR is total_documents
			for docId in documents[term]:
				if vsm_options:
					if docId not in norms:
						norms[docId]=float(token_stats[docId].split(",")[-1].split("|")[1].strip())
					tf = avg_tf_normalization(documents[term][docId][0],float(token_stats[docId].split(",")[0].split("|")[1].strip()))
				else:
					tf = tf_log(documents[term][docId][0]) ## documents[term][docId] is freq. of term in doc of id 'docId'
				tfidf_score = tf * idf
				for field in documents[term][docId][1:]:
					tfidf_score*=field_weights[field]
				tf_idf[docId]+=tfidf_score
	if vsm_options:
		tf_idf = vsm(tf_idf,vsm_options,norms)
	tf_idf =  heapq.nlargest(10,tf_idf.items(),key=operator.itemgetter(1))
	return tf_idf

def get_query_results(path_to_index,tf_idf):
	#doc_title = {}
	results=[]
	for (docId,tfidf_score) in tf_idf:
		docId = int(docId)
		with open(os.path.join(path_to_index,"docid_title_map-" + str(int(docId/10000))), "r") as f:
			for line in f:
				line=line.strip()
				id_ = line.split(":")[0]
				title = ":".join(line.split(":")[1:])
				if id_ == str(docId):
					#doc_title[id_]=title
					results.append(title)
	return results

def is_english(s):
	try:
		s.encode(encoding='utf-8').decode('ascii')
	except UnicodeDecodeError:
		return False
	else:
		return True

def spacy_tokenize(text):
	return [token.text for token in nlp(text,disable = ['ner', 'parser','tagger'])]

def tokenizer(content):
	return spacy_tokenize(content)

def stopwords_removal(tokens):
	return [token for token in tokens if token not in STOP_WORDS]

def punctuations_removal(tokens,type_="str"):
	translator = str.maketrans(string.punctuation + '|', ' '*(len(string.punctuation)+1))
	if type_!="str":
		return [token.translate(translator) for token in tokens]
	else:
		return tokens.translate(translator)

def get_en_lang_tokens(tokens):
	return [token for token in tokens if is_english(token)]

def case_unfolding(tokens,type_="str"):
	if type_!="str":
		return [token.lower() for token in tokens]
	else:
		return tokens.lower()

def strip_text(tokens):
	return [token.strip() for token in tokens if token.strip()]

def stemming(tokens):
	return [porter_stemmer.stem(token) for token in tokens]

def length_check(tokens):
	return [token for token in tokens if len(token)>=2 and len(token)<=10]

def text_normalization(tokens,options):
	if options["case_unfolding"]:
		tokens = case_unfolding(tokens,type_="list")
	if options["length_check"]:
		tokens = length_check(tokens)
	if options["remove_punctuations"]:
		tokens = punctuations_removal(tokens,type_="list")
	if options["strip_tokens"]:
		tokens = strip_text(tokens)
	if options["lang_tokens"]:
		tokens = get_en_lang_tokens(tokens)
	return tokens

def preprocessor(content,normalize_options):
	content=content.strip()
	if content:
		tokens = tokenizer(content)
		normalized_tokens = text_normalization(tokens,normalize_options)
		stopped_tokens = stopwords_removal(normalized_tokens)
		stemmed_tokens = stemming(stopped_tokens)
		return stemmed_tokens
	else:
		return []

def process_text(text):
	options={"case_unfolding":True,"length_check":True,"remove_punctuations":True,"strip_tokens":True,"lang_tokens":True}
	return preprocessor(text,options)

def normal_query(path_to_index,query):
	query = process_text(query)
	if not query:
		return []
	#documents = defaultdict(lambda: defaultdict(int))
	documents = defaultdict(lambda: defaultdict(list))
	fields = ['b', 'c', 'e', 'i', 'r', 't']
	for key in query:
		#temp = defaultdict(int)
		first_char = key[0]
		for field in fields:
			pri_index_file = os.path.join(path_to_index,"index-" + first_char + "-" + field)
			sec_index_file = os.path.join(path_to_index,"secindex-" + first_char + "-" + field)
			if os.path.exists(pri_index_file) and os.path.exists(sec_index_file):
				pri_fd = open(pri_index_file,"r")
				with open(sec_index_file,"r") as sec_fd:
					for line in sec_fd:
						line = line.split(";")
						index_word = line[0]
						if index_word == key:
							for offset in line[1].strip().split(","):
								offset=int(offset)
								pri_fd.seek(offset,0)
								l = pri_fd.readline()
								index_docs = l.split(";")[1].strip().split(",")
								for doc in index_docs:
									doc = doc.split(":")
									docId = doc[0]
									docFreq = int(doc[1])
									if documents[key][docId]:
										documents[key][docId].append(field)
										documents[key][docId][0]+=docFreq
									else:
										documents[key][docId]+=[docFreq,field]
									#documents[key][docId]+=docFreq
							break
	if documents:
		tf_idf = calc_tfidf(documents,vsm_option)
		return get_query_results(path_to_index,tf_idf)
	else:
		return []

def replace_field_names(query):
	for field in field_map:
		query = query.replace(field,field_map[field])
	return query

def get_expanded_query(query):
	query = replace_field_names(query)
	## Assumption:- 1. A field query can only contains query_words associated with valid field tags {b,c,e,i,r,t}
	# with ":" between field tag and query_word(s) . Eg. b:ram , b:ram and shyam etc.
	# 2. Space between ":" and field or between ":" and query word(s) is not allowed
	matched_fields=[]
	for m in query_re.finditer(query):
		matched_fields.append(m.group().split(":")[0].strip())
		query=query.replace(m.group(),"<f>")
	query_words = query.split("<f>")[1:]
	assert len(matched_fields)==len(query_words)

	new_query = []
	new_fields = []
	for (field,word) in zip(matched_fields,query_words):
		processed_tokens = process_text(word.strip())
		if processed_tokens and len(processed_tokens)>0:
			for token in processed_tokens:
				new_query.append(token)
				new_fields.append(field)
	assert len(new_fields)==len(new_query)
	#print(list(zip(new_fields,new_query)))
	return new_fields,new_query

def field_query(path_to_index,query):
	new_fields,new_query = get_expanded_query(query)
	if not new_fields or not new_query:
		return []
	#documents =defaultdict(lambda: defaultdict(int))
	documents =defaultdict(lambda: defaultdict(list))
	for x in range(len(new_query)):
		first_char = new_query[x][0]
		pri_index_file = os.path.join(path_to_index,"index-" + first_char + "-" + new_fields[x])
		sec_index_file = os.path.join(path_to_index,"secindex-" + first_char + "-" + new_fields[x])
		if os.path.exists(pri_index_file) and os.path.exists(sec_index_file):
			pri_fd = open(pri_index_file,"r")
			with open(sec_index_file,"r") as sec_fd:
				for line in sec_fd:
					line = line.split(";")
					index_word = line[0]
					if index_word == new_query[x]:
						for offset in line[1].strip().split(","):
							offset=int(offset)
							pri_fd.seek(offset,0)
							l = pri_fd.readline()
							index_docs = l.split(";")[1].strip().split(",")
							for doc in index_docs:
								doc = doc.split(":")
								docId = doc[0]
								docFreq = int(doc[1])
								#documents[new_query[x]][docId]+=docFreq
								if documents[new_query[x]][docId]:
									documents[new_query[x]][docId].append(new_fields[x])
									documents[new_query[x]][docId][0]+=docFreq
								else:
									documents[new_query[x]][docId]+=[docFreq,new_fields[x]]
						break
	if documents:
		tf_idf = calc_tfidf(documents,vsm_option)
		return get_query_results(path_to_index,tf_idf)
	else:
		return []

def read_file(testfile):
	with open(testfile, 'r') as file:
		queries = file.readlines()
	return queries

def write_file(outputs, path_to_output):
	'''outputs should be a list of lists.
		len(outputs) = number of queries
		Each element in outputs should be a list of titles corresponding to a particular query.'''
	with open(path_to_output, 'w') as file:
		for output in outputs:
			if output:
				for line in output:
					file.write(line.strip() + '\n')
			else:
				file.write("NO-RESULT-FOR-THE-QUERY"+"\n")
			file.write('\n')

def is_field_query(query):
	flag=0
	for field in field_map:
		if field in query:
			flag=1
			break
	return flag

def search(path_to_index, queries):
	'''Write your code here'''
	outputs=[]
	for query in queries:
		query=query.strip()
		if query_re.search(query) or is_field_query(query):
			outputs.append(field_query(path_to_index,query))
		else:
			outputs.append(normal_query(path_to_index,query))
	return outputs


def main():
	path_to_index = sys.argv[1]
	#testfile = sys.argv[2]
	#path_to_output = sys.argv[3]

	#path_to_index="index"
	#testfile="queryfile"
	#path_to_output="resultfile"

	read_token_stats(path_to_index)
	while True:
		try:
			query = input(">>> Enter search query: ")
			if len(query.strip()) < 1:
				sys.exit()
			start_time = time.time()
			outputs = search(path_to_index,[query])
			print("Time(sec) taken to execute query: ",time.time()-start_time)
			print()
			for output in outputs:
				if output:
					for line in output:
						print(line.strip()+"\n")
				else:
					print("NO-RESULT-FOR-THE-QUERY"+"\n")
				print('\n')
		except EOFError:
			sys.exit()
	#queries = read_file(testfile)
	#outputs = search(path_to_index, queries)
	#write_file(outputs, path_to_output)


if __name__ == '__main__':
	main()

