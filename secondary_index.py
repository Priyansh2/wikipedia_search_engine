import os
import re
import sys
import time
from collections import Counter,defaultdict
import bz2
def get_time_info(sec_elapsed):
	h = int(sec_elapsed / (60 * 60))
	m = int((sec_elapsed % (60 * 60)) / 60)
	s = sec_elapsed % 60
	return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def create_secondary_index(primary_index_location,compress_index=True):
	secondary_index_location=primary_index_location
	#secondary_index_location="./sec_index"
	if not os.path.exists(secondary_index_location):
		os.makedirs(secondary_index_location)
	for file in os.listdir(primary_index_location):
		if "index-" in file:
			#print(file)
			if compress_index:
				if not ".bz2" in file:
					continue
			sec_index_file = file.replace("index","secindex")
			sec_index_file = os.path.join(secondary_index_location,sec_index_file)
			if compress_index:
				sec_fd = bz2.open(sec_index_file,"wb")
				f = bz2.open(os.path.join(primary_index_location,file),"rb")
				newline_ch = b'\n'
			else:
				f = open(os.path.join(primary_index_location,file),"r")
				sec_fd = open(sec_index_file,"w")
				newline_ch='\n'
			linecount=0
			sec_index=defaultdict(list)
			cnt=1
			line = f.readline()
			if compress_index:
				line = line.decode("utf-8")
			#if cnt>0:
				#print([line])
			#cnt=-1
			while line:
				#print([line])
				index_word = line.split(";")[0]
				sec_index[index_word].append(str(linecount))
				linecount+=len(line)
				line=f.readline()
				if compress_index:
					line=line.decode("utf-8")
			for word in sec_index:
				content =word+";"+",".join(sec_index[word])
				if compress_index:
					content = content.encode("utf-8")
				sec_fd.write(content+newline_ch)
			sec_fd.close()

def main():
	create_secondary_index(sys.argv[1])
if __name__ == '__main__':
	start_time = time.time()
	main()
	end_time = time.time()
	elapsed_time = end_time - start_time
	print("Elapsed time(sec):",elapsed_time)
	print("Elapsed time(H:M:S): {}".format(get_time_info(elapsed_time)))