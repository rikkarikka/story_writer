import sys
from random import sample
from collections import Counter 


with open("stories.all.ner") as f:
    stories = f.read().strip().split('\n')
with open("titles.all.ner") as f:
    titles = f.read().strip().split('\n')
with open("nv.sents.all") as f:
    nv = f.read().strip().split('\n')

c = Counter(titles)

tests = []
valids = []
for thresh in [5,9]:
  opts = [x for x in c if c[x]>=thresh]
  print(thresh,"+ refs : ",len(opts))

  tests.append(sample(opts,500))
  opts = [x for x in opts if x not in tests[-1]]
  valids.append(sample(opts,500))
  vlen = sum([len(x.split(" ")) for x in valids[-1]])/500
  print("Average title length: ",vlen)

trainf = open("train.all.txt",'w')

skiptitles = []
for i in range(2):
  skiptitles.extend(tests[i])
  skiptitles.extend(valids[i])

skiptitles = set(skiptitles)
trainf.write("\n".join([x+'\t'+stories[i] for i,x in enumerate(titles) if x not in skiptitles]))
trainf.close()
trainidx = open("train.all.idxs",'w')
trainidx.write('\n'.join([str(i) for i,x in enumerate(titles) if x not in skiptitles]))
trainidx.close()

for k,thresh in enumerate([5,9]):
  threshstr = str(thresh)
  vidxf = open("valid."+threshstr+".idxs",'w')
  validf = open("valid."+threshstr+".txt",'w')
  for v in valids[k]:
      idxs = [i for i,x in enumerate(titles) if x == v]
      vidx = sample(idxs,thresh)
      vidxf.write(str(vidx)+'\n')
      validf.write(v+"\t"+"\t".join([stories[i] for i in vidx])+'\n')
  validf.close()

  testf= open("test."+threshstr+".txt",'w')
  for t in tests[k]:
      idxs = [i for i,x in enumerate(titles) if x == t]
      tidx = sample(idxs,thresh)
      testf.write(t+"\t"+"\t".join([stories[i] for i in tidx])+'\n')
  testf.close()

