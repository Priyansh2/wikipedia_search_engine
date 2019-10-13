import os
import re
import sys
import nltk
import string
import time
import itertools
import xml.sax
import xml.sax.handler
import shutil
from tempfile import mkstemp
from shutil import move
from os import fdopen, remove
from collections import defaultdict,Counter
from heapq import heappush, heappop
from nltk.stem.porter import PorterStemmer
from spacy.lang.en.stop_words import STOP_WORDS
from sortedcontainers import SortedDict
from ftfy import fix_text
from unidecode import unidecode
from spacy.lang.en import English

nlp = English()
nlp.max_length = 1000000000
FILES = 0 ## counter of wiki_pages
FILE_CTR = 0 ## file offset of file which contains docid mapped with title
FILE_LIMIT = 10000 ## number of wiki pages to be indexed at once
ARTICLE_MIN_WORDS = 50 ## Ignore article with less than 50 words
FIELD_MIN_TOKENS = 2 ## ignore article if any of its field has less than 2 tokens after preprocessing
DOCID_CTR = 0 ## docId which is mapped to title
DOCID_TITLE_MAP = None ## store file discriptor of docid-title mapping
DOCID_TOKEN_STATS_MAP=None
posting_list = SortedDict()
porter_stemmer = PorterStemmer()
DATA=defaultdict(list)

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

inverted_index="inverted_index" ## temporary/local inverted index folder

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

def create_postings(tokens,elem,docID):
	for stemmed_word in tokens:
		stemmed_word+=elem
		if stemmed_word in posting_list:
			if docID in posting_list[stemmed_word]:
				posting_list[stemmed_word][docID]+=1
			else:
				posting_list[stemmed_word][docID]=1
		else:
			posting_list[stemmed_word]=SortedDict({docID:1})

def create_field_postings(field_token_dict,docID):
	for field in field_token_dict:
		tokens= field_token_dict[field]
		if tokens and len(tokens)>0:
			create_postings(tokens,"-"+field,docID)

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
	options={"case_unfolding":False,"length_check":True,"remove_punctuations":False,"strip_tokens":True,"lang_tokens":True}
	text = case_unfolding(text)
	cited_info,text = extract_cited_info(text)
	infobox,text = extract_infobox(text)
	infobox_tokens =preprocessor(infobox,options)
	#if len(infobox_tokens)<FIELD_MIN_TOKENS:
		#return {}
	category,text = extract_category(text)
	category_tokens = preprocessor(category,options)
	#if len(category_tokens)<FIELD_MIN_TOKENS:
		#return {}

	extlinks,text = extract_extlinks(text)
	extlinks_tokens = preprocessor(extlinks,options)
	#if len(extlinks_tokens)<FIELD_MIN_TOKENS:
		#return {}
	reference,text = extract_references(cited_info,text)
	reference_tokens = preprocessor(reference,options)
	#if len(reference_tokens)<FIELD_MIN_TOKENS:
		#return {}
	text = filter_contents(text)
	bodytext_tokens = preprocessor(text,options)
	#if len(bodytext_tokens)<FIELD_MIN_TOKENS:
		#return {}
	if len(infobox.split())+len(category.split())+len(extlinks.split())+len(reference.split())+len(text.split())>ARTICLE_MIN_WORDS:
		return {"i":infobox_tokens,"c":category_tokens,"e":extlinks_tokens,"r":reference_tokens,"b":bodytext_tokens}
	else:
		return {}

def write_to_file(end=False): ## create partial inverted_index files for handling memory and time issues
	global FILES, FILE_CTR, posting_list
	FILES += 1
	if end or FILES == FILE_LIMIT:
		FILES = 0
		if not os.path.exists(inverted_index):
			os.makedirs(inverted_index)
		with open(inverted_index+"/file" + str(FILE_CTR), "w") as f:
			for x in posting_list:
				new_word = x + ";"
				temp = []
				for y in posting_list[x]:
					temp.append(y + ":" + str(posting_list[x][y]))
				new_word += ",".join(temp) ##stemmed_word-field;doc_id1:freq_w_in_doc1,doc_id2:freq_w_in_doc2. Here freq_w is freq. of stemmed word in field section of the document
				f.write(new_word + "\n")
		FILE_CTR += 1
		posting_list=SortedDict()

def create_secondary_index(primary_index_location):
	secondary_index_location=primary_index_location
	#secondary_index_location="./sec_index"
	if not os.path.exists(secondary_index_location):
		os.makedirs(secondary_index_location)
	for file in os.listdir(primary_index_location):
		if "index-" in file:
			sec_index_file = file.replace("index","secindex")
			sec_index_file = os.path.join(secondary_index_location,sec_index_file)
			sec_fd = open(sec_index_file,"w")
			linecount=0
			sec_index=defaultdict(list)
			with open(os.path.join(primary_index_location,file),"r") as f:
				line = f.readline()
				while line:
					index_word = line.split(";")[0]
					sec_index[index_word].append(str(linecount))
					linecount+=len(line)
					line=f.readline()
			for word in sec_index:
				sec_fd.write(word+";"+",".join(sec_index[word])+"\n")
			sec_fd.close()

def merge_files(remove_index_files=False): ## merging inverted_index files using min_heap to create global index for searching
	index_files = []
	for x in range(FILE_CTR):
		index_files.append(inverted_index+"/file" + str(x))
	open_files = []
	index_heap = [] ## min_heap of index
	for file_ in index_files:
		open_files.append(open(file_,"r"))
	for file_ in open_files:
		line = file_.readline() ## read first line of every index file.
		word = line.split(";")[0] ## word here is: stemmed_word-field
		heappush(index_heap, (word, line, file_)) ## push 3-tuple: (word,line,file_) into empty heap
	prev_filename=''
	prev_fileline=''
	while index_heap:
		smallest = heappop(index_heap)
		first_char = smallest[0][0]
		field = smallest[0].split('-')[1]
		if not os.path.exists(sys.argv[2]):
			os.makedirs(sys.argv[2])
		output_filename =os.path.join(sys.argv[2],'index'+'-'+first_char+'-'+field)
		output_fileline = smallest[1].replace("-" + field, "").strip()
		with open(output_filename, "a") as f:
			f.write(output_fileline+"\n")
		prev_filename = output_filename
		prev_fileline = output_fileline
		next_line = smallest[2].readline()## read next line of index file whose word is smallest (top-most lexicographically sorted word)
		if len(next_line) != 0: ## check if we arrived at end of file, if yes then pop out next smallest word
			word =next_line.split(";")[0]
			heappush(index_heap, (word, next_line, smallest[2]))
	[file_.close() for file_ in open_files]
	if remove_index_files:
		[os.remove(file_) for file_ in index_files]
		shutil.rmtree(inverted_index) ## remove the previously_created partial inverted_index files

def get_time_info(sec_elapsed):
	h = int(sec_elapsed / (60 * 60))
	m = int((sec_elapsed % (60 * 60)) / 60)
	s = sec_elapsed % 60
	return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def get_stats(tokens,options):
	stats=[]
	if options["token_count"]:
		stats.append("token_count|"+str(sum(tokens.values())))
	if options["unique_token_count"]:
		stats.append("unique_token_count|"+str(len(tokens.keys())))
	if options["max_freq_token"]:
		stats.append("max_freq_token|"+str(tokens.most_common()[0][1]))
	if options["avg_token_freq"]:
		stats.append("avg_token_freq|"+str(sum(tokens.values())/len(tokens.keys())))
	if options["min_freq_token"]:
		stats.append("min_freq_token|"+str(tokens.most_common()[-1][1]))
	if options["avg_token_len"]:
		stats.append("avg_token_len|"+str(sum([len(token) for token in tokens.keys()])/len(tokens.keys())))
	if options["doc_len"]:
		stats.append("doc_len|"+str(len(" ".join(tokens))))
	return sorted(stats)

def compute_text_stats(text,stat_options):
	all_tokens=Counter()
	for field in text:
		all_tokens+=Counter(text[field])
	stats = get_stats(all_tokens,stat_options)
	return stats


duplicate_titles=defaultdict(int)
class WikiHandler(xml.sax.handler.ContentHandler):
	def __init__(self):
		self.inTitle=0
		self.inId=0
		self.inText = 0
		self.flag=0
		self.docId=None

	def startElement(self, name, attributes):
		global DOCID_CTR, DOCID_TITLE_MAP,DOCID_TOKEN_STATS_MAP                           #Start Tag
		if name == "id" and self.flag==0:                          #Start Tag: Id
			self.bufferId = ""
			self.inId = 1
			self.flag=1
			if DOCID_CTR%10000==0: ## store the mapping between wiki page (doc_ID) ad title after processing 1000 pages.
				if not os.path.exists(sys.argv[2]):
					os.makedirs(sys.argv[2])
				if DOCID_TITLE_MAP is not None: ## if DOCID_TITLE_MAP file is open then close it, else open it
					DOCID_TITLE_MAP.close()
				if DOCID_TOKEN_STATS_MAP is not None:
					DOCID_TOKEN_STATS_MAP.close()
				DOCID_TOKEN_STATS_MAP = open(os.path.join(sys.argv[2],"docid_token_stats_map-" + str(int(DOCID_CTR/10000))), "w")
				DOCID_TITLE_MAP = open(os.path.join(sys.argv[2],"docid_title_map-" + str(int(DOCID_CTR/10000))), "w")
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
		global DOCID_CTR,DOCID_TOKEN_STATS_MAP,DOCID_TOKEN_STATS_MAP
		if name == "title":
			self.inTitle = 0
		elif name == "text":
			title = fix_text(self.bufferTitle).strip()
			text = fix_text(self.bufferText).strip()
			self.bufferTitle=self.bufferTitle.strip()
			if text and self.bufferTitle and punctuations_removal(self.bufferTitle) and duplicate_titles[self.bufferTitle]<1:
				duplicate_titles[self.bufferTitle]+=1
				text_tokens = process_text(text)
				options={"case_unfolding":True,"length_check":True,"remove_punctuations":True,"strip_tokens":True,"lang_tokens":True}
				if text_tokens:
					text_tokens["t"]=preprocessor(title,options)
					stat_options={"token_count":True,"unique_token_count":True,"max_freq_token":True,"min_freq_token":True,"avg_token_len":True,"avg_token_freq":True,"doc_len":True}
					token_stats = compute_text_stats(text_tokens,stat_options)
					DOCID_TOKEN_STATS_MAP.write(self.docId + ";" + ",".join(token_stats) + "\n")
					create_field_postings(text_tokens,self.docId)
					DOCID_TITLE_MAP.write(self.docId + ":" + self.bufferTitle + "\n")
					write_to_file()
				else:
					DOCID_CTR+=-1
			else:
				DOCID_CTR+=-1
			self.inText = 0
		elif name == "id":
			self.inId = 0
		elif name == "page":
			self.flag=0

def main():
	parser = xml.sax.make_parser()
	parser.setContentHandler(WikiHandler())
	parser.parse(sys.argv[1])
	write_to_file(True) ## to write partial index
	DOCID_TITLE_MAP.close()
	DOCID_TOKEN_STATS_MAP.close()
	merge_files(True)
	create_secondary_index(sys.argv[2])
	with open(os.path.join(sys.argv[2],"docid_ctr"), "w") as f: ## stroing the docID counter (DOCID_CTR) value to file "docid_ctr"
		f.write(str(DOCID_CTR) + "\n")

if __name__ == '__main__':
	start_time = time.time()
	main()
	end_time = time.time()
	elapsed_time = end_time - start_time
	print("Elapsed time(sec):",elapsed_time)
	print("Elapsed time(H:M:S): {}".format(get_time_info(elapsed_time)))