#!/usr/bin/env python
# coding: utf-8

# In[109]:


import math
import os
import time
import re
import sys
import operator
from collections import Counter,defaultdict
from datetime import datetime
from nltk.stem.porter import PorterStemmer


porter_stemmer = PorterStemmer()
query_re = re.compile(r'[t|b|c|e|i|r]:')
camelcase_re = re.compile("([A-Z]+)")
ignore_re = re.compile("[-+*/!\"#$%&\'()=~|^@`\[\]{};:<>,.?_]")
field_map={"title:":"t:","body:":"b:","category:":"c:","ref:":"r:","infobox:":"i:","link:":"e:"}
MAX_WORD_LEN = 10 ## word with length less than or equal to MAX_WORD_LEN is allowed
MIN_WORD_LEN = 3 ## word with length more than or equal to MIN_WORD_LEN is allowed
stop_words = set(['themselves', 'by', 'needn', "wasn't", 'now', 'to', 'why', 'of', 'then', 'after', "needn't",
                  "doesn't", 'all', 'didn', 'just', 'because', 't', 'they', 'most', 'isn', "isn't", 'doesn',
                  'your', 'ain', "shouldn't", 'into', 'and', 'both', 'haven', 'their', "weren't", 'will', 'as',
                  'i', "she's", 'were', "couldn't", 'itself', 'have', 'the', 'its', 'those', 'couldn', 'yours',
                  "mustn't", 'other', "you'll", 'been', 'he', 'don', 'shan', 'is', 'further', 'wasn', 'or',
                  'which', "should've", 'whom', 'does', 'an', 'o', "mightn't", "that'll", 'out', 'up', 'his',
                  'ours', 'so', "you've", 'her', "wouldn't", 'mightn', 'she', 'again', 'him', 'do', "hadn't",
                  'who', 'until', "you're", 'being', 'has', 'through', 'y', 'should', 'myself', 'there', 'with',
                  'what', 'be', 've', 'having', 'yourself', 'we', "didn't", 'hadn', 'had', 'll', 'doing', 'for',
                  'me', 'at', 'can', 'below', 'am', 'here', 'only', 'these', 'not', 'any', 'own', "it's",
                  'himself', 'm', 'a', 'no', 'wouldn', 's', 'won', 'under', 'each', 'them', 'against', 'some',
                  'same', 'in', 'aren', 'mustn', 'are', "you'd", 'about', "shan't", 'theirs', 'was', 'down', 'd',
                  'very', 'how', 'nor', 'ma', 'hasn', "hasn't", "aren't", 're', 'while', "don't", 'above',
                  'yourselves', 'if', 'more', 'hers', 'our', 'once', 'during', 'off', 'over', 'from', 'herself',
                  "won't", 'than', 'before', 'on', 'my', "haven't", 'few', 'between', 'when', 'ourselves', 'that',
                  'did', 'weren', 'too', 'this', 'you', 'shouldn', 'such', 'it', 'where', 'but'])
DOC_CTR = open("docid_ctr","r").read().split("\n")[0].strip()
DOC_CTR=int(DOC_CTR)
#print(DOC_CTR)


# In[110]:


def check_alphanumeric(word):
    ## check if word contains only letters, numbers, underscores and dashes (alphanumeric string)
    if word.isalnum():
        return True
    return False

def check_alphabet(word):
    ## check if word has all characters as alphabets
    if word.isalpha():
        return True
    return False

def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def length_check(word):
    if len(word)>MIN_WORD_LEN and len(word)<MAX_WORD_LEN:
        return True
    return False


def custom_tokenizer(content):
    content = camelcase_re.sub(r' \1', content)
    content = ignore_re.sub(" ", content)
    content = content.replace("\\", " ")
    content = content.split()
    return content

def tokenizer(content):
    return custom_tokenizer(content)


def stopwords_removal_and_stemming(tokens):
    filter_tokens=[]
    if tokens and len(tokens[0].strip())>0:
        for token in tokens:
            token=token.strip().lower()
            if token and token not in stop_words and is_english(token):
                filter_tokens.append(porter_stemmer.stem(token))
    return filter_tokens

def process_text(text):
    tokens = tokenizer(text)
    return stopwords_removal_and_stemming(tokens)

def get_time_info(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def compute_tf(freq):
    return 1 + math.log10(freq)
def compute_idf(num_docs):
    return math.log10(DOC_CTR/num_docs)

def calc_tfidf(documents):
    tf_idf = defaultdict(float) ## it stores tfidf score of retrieved documents for a query
    for term in documents: ## 'term' here means index_word
        num_term_docs = len(documents[term]) ## 'num_term_docs' is number of documents in which word present
        if num_term_docs: ## if num_term_docs is atleast 1, because term's absence from any document will not contribute to search
            idf = math.log10(DOC_CTR/num_term_docs) ## DOC_CTR is total_documents
            for docId in documents[term]:
                tf = 1 + math.log10(documents[term][docId]) ## documents[term][docId] is freq. of term in doc of id 'docId'
                tfidf_score = tf * idf
                tf_idf[docId]+=tfidf_score
    tf_idf = sorted(tf_idf.items(), key=operator.itemgetter(1),reverse=True)[:10] ## top 10 results sorted by tfidf score
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
                if line.count(":")>1:
                    title=":".join(line.split(":")[1:])
                else:
                    title = line.split(":")[1]
                if id_ == str(docId):
                    #doc_title[id_]=title
                    results.append(title)
    return results


# In[111]:


def normal_query(path_to_index,query):
    query = process_text(query)
    #print(query)
    if not query and len(query[0].strip())<1:
        return []
    documents = defaultdict(lambda: defaultdict(int))
    fields = ['b', 'c', 'e', 'i', 'r', 't']
    for key in query:
        #temp = defaultdict(int)
        first_char = key[0]
        for field in fields:
            filename = os.path.join(path_to_index,"index-" + first_char + "-" + field)
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    for line in f:
                        line = line.split(";")
                        index_word = line[0]
                        if index_word== key:
                            index_docs = line[1].strip().split(",")
                            for doc in index_docs:
                                doc = doc.split(":")
                                docId = doc[0]
                                docFreq = int(doc[1])
                                documents[key][docId]+=docFreq
    #print(documents)
    tf_idf = calc_tfidf(documents)
    return get_query_results(path_to_index,tf_idf)

def replace_field_names(query):
    for field in field_map:
        query = query.replace(field,field_map[field])
    return query


def get_expanded_query(query):
    query = replace_field_names(query)
    #print(query)
    ## Assumption:- 1. A field query can only contains query_words associated with valid field tags {b,c,e,i,r,t}
    # with ":" between field tag and query_word(s) . Eg. b:ram , b:ram and shyam etc.
    # 2. Mixed query which contains both field and non-field words are not allowed.
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
        if processed_tokens and len(processed_tokens[0].strip())>0:
            for token in processed_tokens:
                new_query.append(token)
                new_fields.append(field)
    assert len(new_fields)==len(new_query)
    #print(list(zip(new_fields,new_query)))
    return new_fields,new_query

def field_query(path_to_index,query):
    new_fields,new_query = get_expanded_query(query)
    documents =defaultdict(lambda: defaultdict(int))
    for x in range(len(new_query)):
        first_char = new_query[x][0]
        filename = os.path.join(path_to_index,"index-" + first_char + "-" + new_fields[x])
        if os.path.exists(filename):
            with open(filename, "r") as f:
                for line in f:
                    line = line.split(";")
                    index_word = line[0]
                    if index_word == new_query[x]:
                        index_docs = line[1].strip().split(",")
                        for doc in index_docs:
                            doc = doc.split(":")
                            docId = doc[0]
                            docFreq = int(doc[1])
                            documents[new_query[x]][docId]+=docFreq
    tf_idf = calc_tfidf(documents)
    return get_query_results(path_to_index,tf_idf)


# In[112]:


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
            for line in output:
                file.write(line.strip() + '\n')
            file.write('\n')
def is_field_query(query):
    flag=0
    for field in field_map:
        if field in query:
            flag=1
            break
    #print(flag)
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


# In[113]:


def main():
    #path_to_index="global_index"
    #testfile="queryfile"
    #path_to_output="resultfile"
    path_to_index = sys.argv[1]
    testfile = sys.argv[2]
    path_to_output = sys.argv[3]
    queries = read_file(testfile)
    '''queries=["t:gandhi b:arjun i:gandhi c:gandhi r:gandhi",
             "title:gandhi body:arjun infobox:gandhi category:gandhi ref:gandhi",
            "t:mahatma gandhi b:karan arjun i:gandhi nehru patel c:gandhi r:mahatma indra-gandhi",
             "title:mahatma gandhi body:karan arjun infobox:gandhi nehru patel category:gandhi link:mahatma indra-gandhi"]'''
    outputs = search(path_to_index, queries)
    write_file(outputs, path_to_output)


if __name__ == '__main__':
    #start_time = time.time()
    main()
    #end_time = time.time()
    #elapsed_time = end_time - start_time
    #print("Elapsed time(sec):",elapsed_time)
    #print("Elapsed time(H:M:S): {}".format(get_time_info(elapsed_time)))

