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

print(len(nvtrain))
def make_files():
  nouns = []
  verbs = []
  for x in nvtrain:
    nouns.extend([y.split("_")[0].lower() for y in x.split() if "_N" in y])
    verbs.extend([y.split("_")[0].lower() for y in x.split() if "_V" in y])

  nounfreq = Counter(nouns)
  verbfreq = Counter(verbs)
  with open("nouncount.raw",'w') as f:
    for n,c in nounfreq.most_common():
      f.write(n+"\t"+str(c)+'\n')
  with open("verbcount.raw",'w') as f:
    for n,c in verbfreq.most_common():
      f.write(n+"\t"+str(c)+'\n')

  print("nouns: ",len(nouns))
  syns = []

  for n in nouns:
    s = getbestsyn(n)
    if s:
      syns.append(s)

  print(len(syns))
  syns = [x.name() for x in syns]
  syns = set(syns)
  syns = [wn.synset(x) for x in syns]
  print('total syns: ',len(syns))
  #divide into ~500 groups
  maxgroup = 50
  mingroup = 30
  groups = []
  leftover = []
  visited = []
  def breakup(hyper, hypos):
    print(hyper)
    extras = []
    if len(hypos)>mingroup:
      if len(hypos)<maxgroup:
        groups.append((hyper,set(hypos+[hyper])))
      else:
        s = hyper.hyponyms()
        s = [x for x in s if x not in visited]
        visited.extend(s)
        kids = [(k,[x for x in hypos if k in x.lowest_common_hypernyms(k)]) for k in s] 
        kids = [x for x in kids if len(x[1])>0]
        for k,h in kids:
          extras.extend(breakup(k,h))
        if len(extras)>mingroup:
          groups.append((hyper,extras))
          extras = []
        elif hyper in hypos:
          extras.append(hyper)
    else:
      if hyper in hypos:
        extras.append(hyper)
        extras.extend(hypos)
    return extras
    '''
    s = hyper.hyponyms()
    s = [x for x in s if x not in visited]
    visited.extend(s)
    if hyper in hypos:
      leftover.append(hyper)
    kidlens = [len(x) for x in kids]
    smaller = []
    for i,l in enumerate(kidlens):
      if l<mingroup:
        leftover.extend(kids[i])
      elif l>maxgroup:
        breakup(s[i],kids[i])
      else:
        groups.append((s[i],kids[i]))
    if smaller:
      leftover.extend(smaller)
      #groups.append((hyper,smaller))
    '''


  leftover = breakup(wn.synsets("entity")[0],syns)
  groups.append((wn.synsets("entity")[0],leftover))
  print(leftover)

  cats = []
  dic = {}
  for x,y in groups:
    cats.append(x)
    print(x.name(),len(y))
    for s in y:
      dic[s.name()] = x.name()
  with open("noun_categories.txt",'w') as f:
    f.write("\n".join([x.name() for x in cats]))
  with open("noundic.pickle",'wb') as f:
    pickle.dump(dic,f)



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

toocommon = "seem"

def make_data():
  wnl = WNL()
  with open("train.all.idxs") as f:
    tidx = set([int(x) for x in f.read().strip().split('\n')])

  verbd,vcats = make_wordd()

  with open("noundic.pickle",'rb') as f:
    ndic = pickle.load(f)
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
          if not s:
            continue
          s = s.name()
          if s in ndic:
            s = ndic[s]
            nouns.append(s)
          else:
            continue
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
      nouns = [x for x in nouns if x not in newl and x not in toocommon]
      verbs = [x for x in verbs if x not in newl and x not in toocommon]
      if nouns:
        newl.append(nouns[0])
      else:
        newl.append("<oldnoun>")
      if verbs:
        newl.append(verbs[0])
      else:
        newl.append("<oldverb>")

    if i in tidx:
      out.append(" ".join(newl))
      al.append(" ".join(align))
  with open("train.cats",'w') as f:
    f.write("\n".join(out))

if __name__=="__main__":
  if len(sys.argv)>1:
    print('mkfiles')
    make_files()
  else:
    make_data()
