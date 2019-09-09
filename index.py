import os
import re
import sys
import nltk
import string
import time
import itertools
import xml.sax
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
from spacy.lang.en import English

nlp = English()
nlp.max_length = 1000000000
FILES = 0 ## number of wiki_pages to be indexed at once (batch_processing)
FILE_CTR = 0 ## file offset in which partial index is stored
FILE_LIMIT = 10000 ## number of wiki pages to be indexed at once and then saved the posting list to file
DOCID_CTR = 0 ## docId which is mapped to title of wiki_page
DOCID_TITLE_MAP = None ## if not None, docID mapped to wiki title and mapping stored in the file
posting_list = SortedDict() ## posting list which contain field information and docId corresponding to stemmed word
porter_stemmer = PorterStemmer() ## porter stemmer
link_re = re.compile(r'\[(\w+):\/\/(.*?)(( (.*?))|())\]', re.UNICODE) ## find any links
ref_link_re = re.compile(r'\[([^][]*)\|([^][]*)\]', re.DOTALL | re.UNICODE) ## find ref. links
url_re = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+') ## find links start with "http(s)"
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

def punctuations_removal(tokens):
	return [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens]

def get_en_lang_tokens(tokens):
	return [token for token in tokens if is_english(token)]

def case_unfolding(tokens):
	return [token.lower() for token in tokens]

def strip_text(tokens):
	return [token.strip() for token in tokens if token.strip()]

def stemming(tokens):
	return [porter_stemmer.stem(token) for token in tokens]

def length_check(tokens):
	return [token for token in tokens if len(token)>=2 and len(token)<=10]

def text_normalization(tokens):
	lowercased_tokens = case_unfolding(tokens)
	min_len_tokens = length_check(lowercased_tokens)
	punct_removed_tokens = punctuations_removal(min_len_tokens)
	stripped_tokens = strip_text(punct_removed_tokens)
	langspecific_tokens = get_en_lang_tokens(stripped_tokens)
	return langspecific_tokens

def process_content(content,elem,docID):
	content=content.strip()
	if content:
		tokens = tokenizer(content)
		normalized_tokens = text_normalization(tokens)
		stopped_tokens = stopwords_removal(normalized_tokens)
		stemmed_tokens = stemming(stopped_tokens) ## stemming the word of textual content present in "elem" field, to create posting list with key 'stemmed_word-field'
		for stemmed_word in stemmed_tokens:
			stemmed_word+=elem
			if stemmed_word in posting_list:
				if docID in posting_list[stemmed_word]:
					posting_list[stemmed_word][docID]+=1
				else:
					posting_list[stemmed_word][docID]=1
			else:
				posting_list[stemmed_word]=SortedDict({docID:1})


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

def replace(file_path, pattern, subst):
	#Create temp file
	fh, abs_path = mkstemp()
	with fdopen(fh,'w') as new_file:
		with open(file_path) as old_file:
			for line in old_file:
				line=line.strip()
				new_file.write(line.replace(pattern.strip(), subst.strip())+"\n")
	remove(file_path) #Remove original file
	move(abs_path, file_path) #Move new file

def mergeArrays(arr1, arr2):
	## merge sorted arrays arr1 and arr2 in O(n1+n2)
	n1,n2 = len(arr1),len(arr2)
	arr3 = [(None,None)] * (n1 + n2)
	i,j,k = 0,0,0
	while i < n1 and j < n2:
		if arr1[i][0] < arr2[j][0]:
			arr3[k] = arr1[i]
			i+=1
		elif arr1[i][0]>arr2[j][0]:
			arr3[k] = arr2[j]
			j+=1
		else:
			arr3[k] = (arr2[j][0],arr1[i][1]+arr2[j][1])
			j+=1
			i+=1
		k+=1
	while i < n1:
		arr3[k] = arr1[i];
		k+=1
		i+=1
	while j < n2:
		arr3[k] = arr2[j];
		k+=1
		j+=1
	return [x for x in arr3 if (x[0],x[1])!=(None,None)]

def merge_postings(prev_fileline,output_fileline):
	prev_fileline=prev_fileline.strip()
	output_fileline=output_fileline.strip()
	word = output_fileline.split(";")[0]
	l1 = [(int(docid_freq_pair.split(":")[0]),int(docid_freq_pair.split(":")[1])) for docid_freq_pair in prev_fileline.split(";")[1].split(",")]
	l2 = [(int(docid_freq_pair.split(":")[0]),int(docid_freq_pair.split(":")[1])) for docid_freq_pair in output_fileline.split(";")[1].split(",")]
	l = mergeArrays(l1,l2) ## merge in O(n) where n= sum of sizes of posting lists
	merged_posting_list=''
	merged_posting_list+=word+";"+",".join(str(docid)+":"+str(freq) for docid,freq in l)
	return merged_posting_list

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
		'''if output_filename==prev_filename and output_fileline.split(";")[0]==prev_fileline.split(";")[0]:
			output_fileline = merge_postings(prev_fileline,output_fileline)
			replace(os.path.join(os.getcwd(),output_filename.replace("../","").replace("./","")),prev_fileline,output_fileline)
		else:'''
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

def process_title(title):
	caption_re = re.compile(r'\[\[([fF]ile:|[iI]mage)[^]]*(\]\])', re.UNICODE)
	if caption_re.search(title):
		return find_caption(title,caption_re)
	else:
		return title

def find_caption(s,caption_re):
	for match in re.finditer(caption_re, s):
		m = match.group(0)
		caption = m[:-2].split('|')[-1]
		return caption

def find_references(content):
	content = re.sub(link_re, '\\3', content)
	content = re.sub(ref_link_re,'\\2', content)
	content = content.replace('[',"").replace(']',"")
	return content

def find_bodytext(text):
	RE_P12 = re.compile(r'(({\|)|(\|-(?!\d))|(\|}))(.*?)(?=\n)', re.UNICODE)
	RE_P13 = re.compile(r'(?<=(\n[ ])|(\n\n)|([ ]{2})|(.\n)|(.\t))(\||\!)([^[\]\n]*?\|)*', re.UNICODE)
	RE_P17 = re.compile(r'(\n.{0,4}((bgcolor)|(\d{0,1}[ ]?colspan)|(rowspan)|(style=)|(class=)|(align=)|(scope=))(.*))|'r'(^.{0,2}((bgcolor)|(\d{0,1}[ ]?colspan)|(rowspan)|(style=)|(class=)|(align=))(.*))',re.UNICODE)
	text = re.sub(link_re, '\\3', text)
	text = re.sub(ref_link_re,'\\2', text)
	text = url_re.sub(" ", text)
	text = text.replace("!!", "\n|")  # each table head cell on a separate line
	text = text.replace("|-||", "\n|")  # for cases where a cell is filled with '-'
	text = re.sub(RE_P12, '\n', text)  # remove formatting lines
	text = text.replace('|||', '|\n|')  # each table cell on a separate line(where |{{a|b}}||cell-content)
	text = text.replace('||', '\n|')  # each table cell on a separate line
	text = re.sub(RE_P13, '\n', text)  # leave only cell content
	text = re.sub(RE_P17, '\n', text)  # remove formatting lines
	text =text.replace('[',"").replace(']',"")
	return text

def find_linktext(content):
	content = re.sub(link_re, '\\3', content)
	content = re.sub(ref_link_re,'\\2', content)
	content = url_re.sub(" ", content)
	content = content.replace('[',"").replace(']',"")
	return content

def find_categories(content):
	category_re = re.compile("Category:",re.UNICODE)
	content = category_re.sub(" ",content)
	content = re.sub(ref_link_re,'\\2', content)
	content = content.replace('[',"").replace(']',"")
	return content

def get_time_info(sec_elapsed):
	h = int(sec_elapsed / (60 * 60))
	m = int((sec_elapsed % (60 * 60)) / 60)
	s = sec_elapsed % 60
	return "{}:{:>02}:{:>05.2f}".format(h, m, s)

class WikiHandler(xml.sax.ContentHandler):
	def __init__(self):
		self.currElement = None ## to find starting tag
		self.innerElement = None ## to find fields like references, infobox, extlinks, category
		self.docId = None ## docId or wiki_page id/number which holds one-to-one mapping with wiki titles
		self.newLine = 0 ## newline counts. I used it to find external links
		self.title = "" ## to hold title (title buffer)
		self.body = "" ## to hold body text (bodytext buffer)

	def startElement(self, name, attrs):
		global DOCID_CTR, DOCID_TITLE_MAP
		self.currElement = name
		if self.currElement == "page": ## if start tag found is "page" which means it is starting of wiki article
			if DOCID_CTR%10000==0: ## store the mapping between wiki page (doc_ID) ad title after processing 1000 pages.
				if DOCID_TITLE_MAP is not None: ## if DOCID_TITLE_MAP file is open then close it, else open it
					DOCID_TITLE_MAP.close()
				if not os.path.exists(sys.argv[2]):
					os.makedirs(sys.argv[2])
				DOCID_TITLE_MAP = open(os.path.join(sys.argv[2],"docid_title_map-" + str(int(DOCID_CTR/10000))), "w")
			self.docId = str(DOCID_CTR)
			DOCID_CTR += 1
			self.title = ""

		elif self.currElement == "text": ## is start tag found is "text"
			self.body = ""
		self.innerElement = None ## when starting tag is found then at that time we dont have any inner content info like ref., outlinks, etc.

	def characters(self, content):
		if self.currElement == "title":
			content = fix_text(content)
			self.title += content ## write content into title buffer once title tag is found
		elif self.currElement == "text":
			content = fix_text(content)
			if self.innerElement is None: ## this means till now we dont know what content is among links, refs,cats, etc. So we search them.
				if "==External links==" in content or "== External links ==" in  content:
					self.innerElement = "externallinks" ## we found the content to be externallinks
				elif "{{Infobox" in content:
					self.innerElement = "infobox" ## we found the content to be infobox
				elif "Category:" in content:
					process_content(find_categories(content.strip()), "-c", self.docId)
				elif "== References ==" in content or "==References==" in content:
					self.innerElement = "references"
				elif "#REDIRECT" not in content: ## ignore the text content which contains keyword #REDIRECT
					self.body += content

			elif self.innerElement == "externallinks":
				if content == "\n":
					self.newLine += 1
				else:
					self.newLine = 0
				if self.newLine == 2: ## if got two newlines that means we are done with externallinks
					self.newLine = 0
					self.innerElement = None
				process_content(find_linktext(content.strip()), "-e", self.docId)

			elif self.innerElement == "infobox":
				if content == "}}": ## if content is doubly closed curly braces then it means we are done with infobox
					self.innerElement = None
				elif content != "\n" and "=" in content:
					pos = content.index("=") ## find the position of "=" in content
					content = content[pos+1:] ## find the infomation in the box after "="
					process_content(content.strip(), "-i", self.docId)

			elif self.innerElement == "references":
				if content == "\n":
					self.newLine += 1
				else:
					self.newLine = 0
				if self.newLine == 2:
					self.newLine = 0
					self.innerElement = None
				content = content.replace("{{reflist}}","").replace("{{Reflist}}","")
				process_content(find_references(content.strip()), "-r", self.docId)

	def endElement(self, name):
		if self.currElement == "text":
			process_content(find_bodytext(self.body.strip()), "-b", self.docId)
			process_content(process_title(self.title.strip()),"-t", self.docId)
			DOCID_TITLE_MAP.write(self.docId + ":" + self.title.strip() + "\n") ## Eg: docId:wiki_title
			write_to_file()

def main():
	parser = xml.sax.make_parser()
	parser.setContentHandler(WikiHandler())
	parser.parse(open(sys.argv[1],"r"))
	write_to_file(True) ## to write partial index
	DOCID_TITLE_MAP.close()
	merge_files(True)
	with open("docid_ctr", "w") as f: ## stroing the docID counter (DOCID_CTR) value to file "docid_ctr"
		f.write(str(DOCID_CTR) + "\n")

if __name__ == '__main__':
	start_time = time.time()
	main()
	end_time = time.time()
	elapsed_time = end_time - start_time
	print("Elapsed time(sec):",elapsed_time)
	print("Elapsed time(H:M:S): {}".format(get_time_info(elapsed_time)))