import sys
import pickle
from collections import Counter
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer as WNL


def make_wordd():
  wnl = WNL()
  with open("verb_cats.txt") as f:
    cats = f.read().strip().split('\n')
  wordassoc = {}
  with open("verb_vocab.txt") as f:
    ctr = 0
    for l in f:
      l = l.strip()
      if l == "___":
        ctr+=1
      else:
        if l not in wordassoc:
          if l not in ['was','had']:#'is','be','been','being','get','got','were','had','has','have','are','will']:
            wordassoc[l] = ctr

  print(len(wordassoc))
  return wordassoc,cats

with open("nounfreq.cats") as f:
  nouns = [x.split('\t')[0] for x in f.read().strip().split("\n")]
with open("verbfreq.cats") as f:
  verbs = [x.split('\t')[0] for x in f.read().strip().split("\n")]

_, cats = make_wordd()
rarest = []
freq = []
with open("t_train.cats") as f:
  for l in f:
    l = l.split()
    sl = set(l)
    ns = [(nouns.index(x),x) for x in sl if not x.isdigit()]
    vs = [(verbs.index(cats[int(x)]),cats[int(x)]) for x in sl if x.isdigit()]
    ns.sort(reverse=True)
    vs.sort(reverse=True)
    rarest.append([x[1] for x in ns]+[";"]+[x[1] for x in vs])
    n = [x for x in l if not x.isdigit()]
    v = [x for x in l if x.isdigit()]
    nc = Counter(n)
    vc = Counter(v)
    ftmp = []
    for i in set(nc.values()):
      order = [(l.index(x),x) for x in nc if nc[x]==i]
      order.sort()
      ftmp.extend([x[1] for x in order])
    ftmp.append(";")
    for i in set(vc.values()):
      order = [(l.index(x),cats[int(x)]) for x in vc if vc[x]==i]
      order.sort()
      ftmp.extend([x[1] for x in order])
    freq.append(ftmp)

with open("t_train_rare.cats",'w') as f:
  f.write("\n".join([" ".join(x) for x in rarest]))
with open("t_train_freq.cats",'w') as f:
  f.write("\n".join([" ".join(x) for x in freq]))



rarest = []
freq = []
with open("t_val.cats") as f:
  for l in f:
    l = l.split()
    sl = set(l)
    ns = [(nouns.index(x),x) for x in sl if not x.isdigit()]
    vs = [(verbs.index(cats[int(x)]),cats[int(x)]) for x in sl if x.isdigit()]
    ns.sort(reverse=True)
    vs.sort(reverse=True)
    rarest.append([x[1] for x in ns]+[";"]+[x[1] for x in vs])
    n = [x for x in l if not x.isdigit()]
    v = [x for x in l if x.isdigit()]
    nc = Counter(n)
    vc = Counter(v)
    ftmp = []
    for i in set(nc.values()):
      order = [(l.index(x),x) for x in nc if nc[x]==i]
      order.sort()
      ftmp.extend([x[1] for x in order])
    ftmp.append(";")
    for i in set(vc.values()):
      order = [(l.index(x),cats[int(x)]) for x in vc if vc[x]==i]
      order.sort()
      ftmp.extend([x[1] for x in order])
    freq.append(ftmp)

with open("t_val_rare.cats",'w') as f:
  f.write("\n".join([" ".join(x) for x in rarest]))
with open("t_val_freq.cats",'w') as f:
  f.write("\n".join([" ".join(x) for x in freq]))



    

