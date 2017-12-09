import sys
from random import sample
from collections import Counter 

threshstr = sys.argv[1]
thresh = int(threshstr)

with open("stories.all.ner") as f:
    stories = f.read().strip().split('\n')
with open("titles.all.ner") as f:
    titles = f.read().strip().split('\n')
with open("nv.sents.all") as f:
    nv = f.read().strip().split('\n')

c = Counter(titles)
opts = [x for x in c if c[x]>=thresh]
print(thresh,"+ refs : ",len(opts))

test = sample(opts,500)
opts = [x for x in opts if x not in test]
valid = sample(opts,500)
vlen = sum([len(x.split(" ")) for x in valid])/500
print("Average title length: ",vlen)

trainf = open("train."+threshstr+".txt",'w')

skipidx = [i for i,x in enumerate(titles) if x in valid or x in test]
trainf.write("\n".join([x+'\t'+stories[i] for i,x in enumerate(titles) if i not in skipidx]))
trainf.close()
trainidx = open("train."+threshstr+".idxs",'w')
trainidx.write('\n'.join([str(i) for i in range(len(titles)) if i not in skipidx]))
trainidx.close()

vidxf = open("valid."+threshstr+".idxs",'w')
validf = open("valid."+threshstr+".txt",'w')
for v in valid:
    idxs = [i for i,x in enumerate(titles) if x == v]
    vidx = sample(idxs,thresh)
    vidxf.write(str(vidx)+'\n')
    validf.write(v+"\t"+"\t".join([stories[i] for i in vidx])+'\n')
validf.close()

testf= open("test."+threshstr+".txt",'w')
for t in test:
    idxs = [i for i,x in enumerate(titles) if x == t]
    tidx = sample(idxs,thresh)
    testf.write(t+"\t"+"\t".join([stories[i] for i in tidx])+'\n')
testf.close()

