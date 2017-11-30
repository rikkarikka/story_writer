import sys
import pickle
from nltk.corpus import wordnet as wn

with open("nouns.train") as f:
  data = f.readlines()
  #data = data[:1000]
  nouns = set(" ".join(data).split())

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

print("nouns: ",len(nouns))
try:
  with open("noundic.pickle",'rb') as f:
    dic = pickle.load(f)
except:
  syns = []

  for n in nouns:
    s = getbestsyn(n)
    if s:
      syns.append(s)

  print(len(syns))
  syns = [x.name() for x in syns]
  syns = set(syns)
  syns = [wn.synset(x) for x in syns]
  print(len(syns))
  #divide into ~500 groups
  maxgroup = 50
  mingroup = 30
  groups = []
  leftover = []
  visited = []
  def breakup(hyper, hypos):
    s = hyper.hyponyms()
    s = [x for x in s if x not in visited]
    visited.extend(s)
    if hyper in hypos:
      leftover.append(hyper)
    kids = [[x for x in hypos if k in x.lowest_common_hypernyms(k)] for k in s] 
    kidlens = [len(x) for x in kids]
    smaller = []
    for i,l in enumerate(kidlens):
      if l<mingroup:
        smaller.extend(kids[i])
      elif l>maxgroup:
        breakup(s[i],kids[i])
      else:
        groups.append((s[i],kids[i]))
    if smaller:
      groups.append((hyper,smaller))
      


  breakup(wn.synsets("entity")[0],syns)
  groups.append((wn.synsets("entity")[0],leftover))

  cats = []
  dic = {}
  names = [s.name() for s in syns]
  for x,y in groups:
    cats.append(x)
    print(x.name(),len(y))
    for s in y:
      dic[s.name()] = x.name()
  with open("noun_categories.txt",'w') as f:
    f.write("\n".join([x.name() for x in cats]))
  with open("noundic.pickle",'wb') as f:
    pickle.dump(dic,f)

  
out = []
for l in data:
  l = l.strip().split()
  newl = []
  for n in l:
    s = getbestsyn(n)
    if not s: continue
    s = s.name()
    if s in dic:
      newl.append(n+"->"+dic[s])
  out.append(" ".join(newl))
with open("nouns2cats.train",'w') as f:
  f.write("\n".join(out))
