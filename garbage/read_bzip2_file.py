import bz2
import os,re,sys
from collections import defaultdict,Counter

filename ="index-0-t.bz2"
with bz2.open(filename,"rb") as f:
	content = f.read().decode("utf-8").split("\n")

first_line  =content[0]
#print(first_line)
last_line = content[-1]
second_last_line = content[-2]
print([first_line],[last_line],[second_last_line])

newfile = "test.bz2"
with bz2.open(newfile,"wb") as f:
	f.write(second_last_line.encode("utf-8")+b'\n')