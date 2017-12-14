import sys
import pickle
from collections import Counter
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer as WNL


def getbestsyn(n):
  s = [x for x in wn.synsets(n) if x.pos()=='n']
  overlap = [x for x in s if n in x.name() or x.name().split('.')[0] in n]
  if overlap: s = overlap
  first = [x for x in s if '01' in x.name()]
  if first: s = first
  if len(s)>1:
    s = sorted([(x.name()[::-1],x) for x in s])
    s = s[0][1]
  elif len(s)==1:
    s = s[0]
  return s

with open('train.all.idxs') as f:
  tidxs = set([int(x) for x in f.read().strip().split('\n')])

with open("nv.sents.all") as f:
  nv = f.read().strip().split("\n")
  nvtrain = [x for i,x in enumerate(nv) if i in tidxs]

toocommon = []
wnl = WNL()

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
          wordassoc[l] = ctr

  print(len(wordassoc))
  return wordassoc,cats

def make_data():
  wnl = WNL()
  verbd,vcats = make_wordd()
  with open("train.all.idxs") as f:
    tidx = set([int(x) for x in f.read().strip().split('\n')])
  hyper = lambda x: x.hypernyms()

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
          s = getbestsyn(n)
          if s:
            close = list(s.closure(hyper))
            if len(close)>1:
              cat = close[1]
            elif len(close)==1:
              cat = close[-1]
            else:
              continue
            nouns.append(cat.name())
        elif pos[0]=='v':
          lem = wnl.lemmatize(n,pos='v')
          if lem in verbd:
            s = verbd[lem]
            s = vcats[s]
            verbs.append(s)
          else:
            continue
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
  with open("train.martin",'w') as f:
    f.write("\n".join(out))

if __name__=="__main__":
  make_data()
