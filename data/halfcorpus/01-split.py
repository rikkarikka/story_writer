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

s = open("valid.idx",'w')
for i in idxs[:m]:
  s.write(str(i)+'\n')
  s.write(src[i]+"\t"+tgt[i])
s.close()

s = open("test.idx",'w')
for i in idxs[m:m*2]:
  s.write(src[i]+"\t"+tgt[i])
s.close()

s = open("train.idx",'w')
for i in idxs[m*2:]:
  s.write(src[i]+"\t"+tgt[i])
s.close()
