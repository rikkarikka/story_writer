import sys
from random import sample
from collections import Counter 

with open("stories.all.ner") as f:
    stories = f.read().strip().split('\n')
with open("titles.all.ner") as f:
    titles = f.read().strip().split('\n')
with open("nv.all") as f:
    nv = f.read().strip().split('\n')

c = Counter(titles)
opts = [x for x in c if c[x]>8]
print("9+ refs : ",len(opts))

test = sample(opts,500)
opts = [x for x in opts if x not in test]
valid = sample(opts,500)

trainf = open("train.txt.ner",'w')

skipidx = [i for i,x in enumerate(titles) if x in valid or x in test]
trainf.write("\n".join([x+'\t'+stories[i] for i,x in enumerate(titles) if i not in skipidx]))
trainf.close()
trainidx = open("train.idxs",'w')
trainidx.write('\n'.join([str(i) for i in range(len(titles)) if i not in skipidx]))
trainidx.close()

validf = open("valid.txt.ner",'w')
for v in valid:
    idxs = [i for i,x in enumerate(titles) if x == v]
    print(idxs)
    vidx = sample(idxs,9)
    validf.write(v+"\t"+"\t".join([stories[i] for i in vidx])+'\n')
validf.close()

testf= open("test.txt.ner",'w')
for t in test:
    idxs = [i for i,x in enumerate(titles) if x == t]
    tidx = sample(idxs,9)
    testf.write(t+"\t"+"\t".join([stories[i] for i in tidx])+'\n')
testf.close()

