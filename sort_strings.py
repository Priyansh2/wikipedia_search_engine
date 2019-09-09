import os
files=[]
for file in os.listdir("index"):
	if "index" in file:
		files.append(os.path.join("index",file))
files=sorted(files)
#file =files[0]
for file in files:
	words=[line.strip().split(";")[0] for line in open(file,"r").read().split("\n") if line.strip()]
	temp = sorted(words,key=lambda s:s.lower())
	assert temp==words
	for i in range(1,len(words)):
		assert words[i]>=words[i-1]