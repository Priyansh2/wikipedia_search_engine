import os
import re
import sys
import time
from collections import Counter,defaultdict

def get_time_info(sec_elapsed):
	h = int(sec_elapsed / (60 * 60))
	m = int((sec_elapsed % (60 * 60)) / 60)
	s = sec_elapsed % 60
	return "{}:{:>02}:{:>05.2f}".format(h, m, s)

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

def main():
	create_secondary_index(sys.argv[1])
if __name__ == '__main__':
	start_time = time.time()
	main()
	end_time = time.time()
	elapsed_time = end_time - start_time
	print("Elapsed time(sec):",elapsed_time)
	print("Elapsed time(H:M:S): {}".format(get_time_info(elapsed_time)))