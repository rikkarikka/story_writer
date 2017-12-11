import sys
import pickle
from collections import Counter
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer as WNL


with open('train.all.idxs') as f:
  tidxs = set([int(x) for x in f.read().strip().split('\n')])

with open("nv.sents.all") as f:
  nv = f.read().strip().split("\n")
  nvtrain = [x for i,x in enumerate(nv) if i in tidxs]

toocommon = []
wnl = WNL()

def make_data():
  wnl = WNL()
  with open("train.all.idxs") as f:
    tidx = set([int(x) for x in f.read().strip().split('\n')])

  out = []
  al = []
  for i,l in enumerate(nv):
    if i not in tidx: continue
    l = l.strip().split("_.")[:-1]
    newl = []
    align = []
    for s in l:
      s = [x for x in s.lower().split(" ")if "_" in x]
      nouns = []
      verbs = []
      for w in s:
        n,pos = w.lower().split("_")
        if pos[0]=='n':
          lem = wnl.lemmatize(n,pos='n')
          nouns.append(lem)
        elif pos[0]=='v':
          lem = wnl.lemmatize(n,pos='v')
          verbs.append(lem)
        else:
          continue
      nounss = [x for x in nouns if x not in newl and x not in toocommon]
      verbss = [x for x in verbs if x not in newl and x not in toocommon]
      if not nouns:
        newl.append("<nonoun>")
      elif nounss:
        newl.append(nounss[0])
      else:
        newl.append("<oldnoun>")
      if not verbs:
        newl.append("<noverb>")
      elif verbss:
        newl.append(verbss[0])
      else:
        newl.append("<oldverb>")

    if i in tidx:
      out.append(" ".join(newl))
      al.append(" ".join(align))
  with open("train.lems",'w') as f:
    f.write("\n".join(out))

if __name__=="__main__":
  make_data()
