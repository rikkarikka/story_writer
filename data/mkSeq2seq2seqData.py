import sys
from collections import Counter, defaultdict
from random import sample

titles = []
txts = []
nouns = []
verbs = []


counts = defaultdict(int)

with open("train.txt.ner") as f:
  for l in f:
    l = l.strip()
    title, txt = l.split("\t")
    titles.append(title)
    txts.append(txt)

ran = list(range(len(titles)))
vidx = sample(ran, 1000)
tidx = [x for x in ran if x not in vidx]

with open("verbs2cats.train") as f:
  for i,l in enumerate(f.read().strip().split('\n')):
    if not l:
      if i in tidx: tidx.remove(i)
      if i in vidx: vidx.remove(i)
      verbs.append([])
      continue
    verb = [x.split("->")[1]+"_V" for x in l.strip().split(" ")]
    c = Counter(verb)
    if i in tidx:
      for k,v in c.items(): counts[k]+=v
    verbs.append(c.most_common())

with open("nouns2cats.train") as f:
  for i,l in enumerate(f.read().strip().split('\n')):
    if not l:
      if i in tidx: tidx.remove(i)
      if i in vidx: vidx.remove(i)
      nouns.append([])
      continue
    noun = [x.split("->")[1]+"_N" for x in l.strip().split(" ")]
    c = Counter(noun)
    if i in tidx:
      for k,v in c.items(): counts[k]+=v
    nouns.append(c.most_common())


seq1 = open("seq2seq2seq/train.seq1",'w')
seq2 = open("seq2seq2seq/train.seq2",'w')
data = []
for i in tidx:
  items = nouns[i]+verbs[i]
  print(items)
  items = [(x[0],float(str(x[1])+"."+str(1000000-counts[x[0]]))) for x in items]
  items.sort(key=lambda x: x[1],reverse=True)
  itemstr = ' '.join([x[0] for x in items])
  seq1.write(titles[i]+"\t"+itemstr+"\n")
  seq2.write(itemstr+"\t"+txts[i]+"\n")
seq1.close()
seq2.close()

seq1 = open("seq2seq2seq/valid.seq1",'w')
seq2 = open("seq2seq2seq/valid.seq2",'w')
data = []
for i in vidx:
  items = nouns[i]+verbs[i]
  items = [(x[0],float(str(x[1])+"."+str(1000000-counts[x[0]]))) for x in items]
  items.sort(key=lambda x: x[1],reverse=True)
  itemstr = ' '.join([x[0] for x in items])
  seq1.write(titles[i]+"\t"+itemstr+"\n")
  seq2.write(itemstr+"\t"+txts[i]+"\n")
seq1.close()
seq2.close()

