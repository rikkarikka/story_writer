import sys
from collections import defaultdict

with open("categories.txt") as f:
  cats = f.read().strip().split('\n')

wordassoc = defaultdict(list)
with open("vocabulary.txt") as f:
  ctr = 0
  for l in f:
    l = l.strip()
    if l == "___":
      ctr+=1
    else:
      wordassoc[l].append(ctr)

lens = []
for fn in ['train.txt.ner.phrases']:#,'valid.txt.phrases']:
  new = open(fn+".pretrain",'w')
  with open(fn) as f:
    for l in f:
      title, txt = l.split('\t')
      txt = [x for x in txt.strip().split() if x in wordassoc]
      tmp = []
      for w in txt:
        tmp.extend([cats[i] for i in wordassoc[w]])
      if tmp:
        #tmp = list(set(tmp))
        lens.append(len(set(tmp)))
        txt = ' '.join(tmp)
        new.write(title+'\t'+txt+'\n')
  new.close()
print(sum(lens)/len(lens))

