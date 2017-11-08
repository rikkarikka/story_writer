import sys
from random import shuffle

with open(sys.argv[1]) as f:
  src = [x.strip() for x in f.readlines()]

with open(sys.argv[2]) as f:
  tgt = f.readlines()

assert(len(src)==len(tgt))

idxs = list(range(len(src)))
shuffle(idxs)

m = 1000 #len(idxs)//10

s = open("valid.txt",'w')
for i in idxs[:m]:
  s.write(src[i]+"\t"+tgt[i])
s.close()

s = open("test.txt",'w')
for i in idxs[m:m*2]:
  s.write(src[i]+"\t"+tgt[i])
s.close()

s = open("train.txt",'w')
for i in idxs[m*2:]:
  s.write(src[i]+"\t"+tgt[i])
s.close()
