import xml.sax.handler
import os
import re
import nltk
import string
import time
import itertools
import shutil
import sys
import heapq
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from sortedcontainers import SortedDict
from ftfy import fix_text
from nltk.stem.porter import PorterStemmer
from collections import defaultdict,Counter
nlp = English()
nlp.max_length = 1000000000
DOCID_CTR = 0 ## docId which is mapped to title
DOCID_TOKEN_STATS_MAP=None
porter_stemmer = PorterStemmer()
links1_re = re.compile(r'\[(\w+):\/\/(.*?)(( (.*?))|())\]', re.UNICODE) ## find any links
links2_re = re.compile(r'\[([^][]*)\|([^][]*)\]', re.DOTALL | re.UNICODE) ## find links embedded inside template
url_re = re.compile('https?:\/\/[^\s\|]+',re.UNICODE) ## find "http(s)" links
category_re = re.compile('\[\[category:([^\]}]+)\]\]',re.UNICODE) ## find categories
extlinks_re = re.compile("==\s?external links\s?==(.*?)\n\n", re.DOTALL | re.UNICODE) ## find 'external links' section
cite_re = re.compile("{{cite?(?:ation)?(.*?)}}",re.DOTALL | re.UNICODE) ## find citations
references1_re=re.compile("<ref((?:[^<])*?)\/>",re.UNICODE) ## pattern1 to find  reference embedded inside template
references2_re=re.compile("<ref((?:[^<])*?)<\/ref>",re.UNICODE) ## pattern2 to find reference embedded inside template
notes_and_references_re = re.compile("==\s?notes and references\s?==(.*?)\n\n", re.DOTALL | re.UNICODE) ## find 'notes and reference' section
further_reading_re = re.compile("==\s?further reading\s?==(.*?)\n\n", re.DOTALL | re.UNICODE) ## find 'futher reading' section
see_also_re = re.compile("==\s?see also\s?==(.*?)\n\n", re.DOTALL | re.UNICODE) ## find 'see also' section

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


def filter_contents(text):
	text = re.sub(url_re,"", text)
	text = re.sub(links1_re, '\\3', text)
	text = re.sub(links2_re,'\\2', text)
	text = re.sub("<blockquote.*?>(.*?)</blockquote>", r"\1 ",re.sub("\n", "", text))
	text = re.sub("{{verify.*?}}", " ",text.rstrip(), re.DOTALL)
	text = re.sub("{{failed.*?}}", " ",text, re.DOTALL)
	text = re.sub("{{page.*?}}", " ",text, re.DOTALL)
	text = re.sub("{{lang.*?fa.*?}}", " ", text, re.DOTALL)
	text = re.sub("{{spaced ndash}}", " ", text, re.DOTALL)
	text = re.sub("{{quote.*?\|(.*?)}}", r"\1 ", text, re.DOTALL)
	text = re.sub("{{main.*?\|(.*?)}}", r"\1 ", text, re.DOTALL)
	text = re.sub("file:.*?\|", " ", text, re.DOTALL)
	text = re.sub("<!-*(.*?)-*>", r"\1 ", text ,re.DOTALL)
	text = punctuations_removal(text)
	return text

def extract_extlinks(text):
	extlinks = filter_contents(" ".join(extlinks_re.findall(text,re.DOTALL)))
	text =re.sub(extlinks_re," ", text)
	return extlinks,text

def extract_category(text):
	text = re.sub(url_re,"", text)
	category = filter_contents(" ".join(category_re.findall(text)))
	text = re.sub(category_re," ",text)
	return category,text

def extract_cited_info(text):
	cited_info=[]
	cites = cite_re.findall(text)
	text = re.sub(cite_re,"",text)
	refs = references1_re.findall(text)
	text = re.sub(references1_re,"",text)
	refs+=references2_re.findall(text)
	refs = filter_contents(" ".join(refs))
	text = re.sub(references2_re,"",text)
	cited_info.append(cites)
	cited_info.append(refs)
	return cited_info,text

def extract_references(cited_info,text):
	cites = cited_info[0]
	refs = cited_info[1]
	notes_and_refs = filter_contents(" ".join(notes_and_references_re.findall(text, re.DOTALL)))
	text = re.sub(notes_and_references_re," ",text)
	further_read = filter_contents(" ".join(further_reading_re.findall(text,re.DOTALL)))
	text = re.sub(further_reading_re," ",text)
	see_also = filter_contents(" ".join(see_also_re.findall(text,re.DOTALL)))
	text = re.sub(see_also_re, " ", text)
	citations=''
	for x in cites:
		for y in x.split('|'):
			if re.search("title", y):
				try:
					citations+= y.split('=')[1]+" "
				except:
					pass
	citations = filter_contents(citations)
	reference_info=" ".join([citations,refs,notes_and_refs,further_read,see_also])
	return reference_info,text

def extract_infobox(text):
	infobox=[]
	for match in reversed(list(re.finditer("{{infobox", text))):
		start=match.span()[0]
		end=start+2
		flag=2
		content=""
		for ch in text[start+2:]:
			end+=1
			if flag==0:
				break
			if ch=="{":
				flag+=1
			elif ch=="}":
				flag+=- 1
			else:
				content+=ch
		text=text[:start]+text[end:]
		infobox.append(content)
	infobox = " ".join(infobox)
	infobox_info=''
	for line in infobox.split("|"):
		try:
			infobox_info+=line.split("=")[1]+" "
		except:
			infobox_info+=line+" "
	infobox_info = filter_contents(infobox_info)
	infobox_info = infobox_info.replace("infobox","")
	return infobox_info,text

def process_text(text):
	## processing each wiki_page parts by parts
	text = case_unfolding(text)
	cited_info,text = extract_cited_info(text)
	infobox,text = extract_infobox(text)
	category,text = extract_category(text)
	extlinks,text = extract_extlinks(text)
	reference,text = extract_references(cited_info,text)
	text = filter_contents(text)
	options={"case_unfolding":False,"length_check":True,"remove_punctuations":False,"strip_tokens":True,"lang_tokens":True}

	return {"i":preprocessor(infobox,options),"c":preprocessor(category,options),"e":preprocessor(extlinks,options),"r":preprocessor(reference,options),"b":preprocessor(text,options)}


def get_stats(tokens,options):
	stats=[]
	if options["token_count"]:
		try:
			token_count = sum(tokens.values())
		except:
			token_count=1
		stats.append("token_count|"+str(token_count))

	if options["unique_token_count"]:
		try:
			unique_token_count = len(tokens.keys())
		except:
			unique_token_count=1
		stats.append("unique_token_count|"+str(unique_token_count))

	if options["max_freq_token"]:
		try:
			max_freq_token = tokens.most_common()[0][1]
		except:
			max_freq_token=1
		stats.append("max_freq_token|"+str(max_freq_token))

	if options["avg_token_freq"]:
		try:
			avg_token_freq = sum(tokens.values())/len(tokens.keys())
		except:
			avg_token_freq=1
		stats.append("avg_token_freq|"+str(avg_token_freq))

	if options["min_freq_token"]:
		try:
			min_freq_token = tokens.most_common()[-1][1]
		except:
			min_freq_token=1
		stats.append("min_freq_token|"+str(min_freq_token))

	if options["avg_token_len"]:
		try:
			avg_token_len = sum([len(token) for token in tokens.keys()])/len(tokens.keys())
		except:
			avg_token_len = 1
		stats.append("avg_token_len|"+str(avg_token_len))

	if options["doc_len"]:
		try:
			doc_len = len(" ".join(tokens))
		except:
			doc_len=1
		stats.append("doc_len|"+str(doc_len))

	return sorted(stats)

def compute_text_stats(text,stat_options):
	all_tokens=Counter()
	for field in text:
		all_tokens+=Counter(text[field])
	stats = get_stats(all_tokens,stat_options)
	return stats

def get_time_info(sec_elapsed):
	h = int(sec_elapsed / (60 * 60))
	m = int((sec_elapsed % (60 * 60)) / 60)
	s = sec_elapsed % 60
	return "{}:{:>02}:{:>05.2f}".format(h, m, s)


class WikiHandler(xml.sax.handler.ContentHandler):
	def __init__(self):
		self.inTitle=0
		self.inId=0
		self.inText = 0
		self.flag=0
		self.docId=None

	def startElement(self, name, attributes):
		global DOCID_CTR,DOCID_TOKEN_STATS_MAP
		if name == "id" and self.flag==0:
			self.bufferId = ""
			self.inId = 1
			self.flag=1
			if DOCID_CTR%10000==0: ## store the mapping between wiki page (doc_ID) ad title after processing 1000 pages.
				if DOCID_TOKEN_STATS_MAP is not None: ## if DOCID_TITLE_MAP file is open then close it, else open it
					DOCID_TOKEN_STATS_MAP.close()
				if not os.path.exists(sys.argv[2]):
					os.makedirs(sys.argv[2])
				DOCID_TOKEN_STATS_MAP = open(os.path.join(sys.argv[2],"docid_token_stats_map-" + str(int(DOCID_CTR/10000))), "w")
			self.docId = str(DOCID_CTR)
			DOCID_CTR += 1

		elif name == "title":
			self.bufferTitle = ""
			self.inTitle = 1
		elif name =="text":
			self.bufferText = ""
			self.inText = 1

	def characters(self, data):
		if self.inId and self.flag==1:
			self.bufferId += data
		elif self.inTitle:
			self.bufferTitle += data
		elif self.inText:
			self.bufferText += data

	def endElement(self, name):
		if name == "title":
			self.inTitle = 0
		elif name == "text":
			title = fix_text(self.bufferTitle)
			text = fix_text(self.bufferText)
			text = process_text(text)
			options={"case_unfolding":True,"length_check":True,"remove_punctuations":True,"strip_tokens":True,"lang_tokens":True}
			text["t"]=preprocessor(title.strip(),options)
			stat_options={"token_count":True,"unique_token_count":True,"max_freq_token":True,"min_freq_token":True,"avg_token_len":True,"avg_token_freq":True,"doc_len":True}
			token_stats = compute_text_stats(text,stat_options)
			DOCID_TOKEN_STATS_MAP.write(self.docId + ";" + ",".join(token_stats) + "\n")
			self.inText = 0
		elif name == "id":
			self.inId = 0
		elif name == "page":
			self.flag=0

def main():
	parser = xml.sax.make_parser()
	parser.setContentHandler(WikiHandler())
	parser.parse(sys.argv[1])
	DOCID_TOKEN_STATS_MAP.close()
if __name__ == '__main__':
	start_time = time.time()
	main()
	end_time = time.time()
	elapsed_time = end_time - start_time
	print("Elapsed time(sec):",elapsed_time)
	print("Elapsed time(H:M:S): {}".format(get_time_info(elapsed_time)))