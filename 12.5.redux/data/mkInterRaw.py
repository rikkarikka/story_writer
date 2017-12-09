import os
import argparse
from nltk.stem import WordNetLemmatizer as WNL
wnl = WNL()
with open("nouncount.raw") as f:
  nouns = [x.split("\t")[0] for x in f.readlines()]
print(len(nouns))
nouns = [wnl.lemmatize(n,pos='n') for n in nouns]
print(len(set(nouns)))
exit()

def parseParams():
  parser = argparse.ArgumentParser(description='none')
  parser.add_argument('-hn', type=int, default=5000)
  parser.add_argument('-taken', type=int, default=500)
  parser.add_argument('-hv', type=int, default=10000)
  parser.add_argument('-takev', type=int, default=250)
  parser.add_argument('-nouns', type=str, default="nouncount.raw")
  parser.add_argument('-verbs', type=str, default="verbcount.raw")
  parser.add_argument('-out', type=str, default="train.raw")
  args= parser.parse_args()
  return args

def getvocab(args):
  nouns = []
  with open(args.nouns) as f:
    i = 0
    while i<args.taken:
      l = next(f)
      w,k = l.split('\t')
      k = int(k)
      if k<args.hn:
        nouns.append(w)
        i+=1
  verbs = []
  with open(args.verbs) as f:
    i = 0
    while i<args.takev:
      l = next(f)
      w,k = l.split('\t')
      k = int(k)
      if k<args.hv:
        verbs.append(w)
        i+=1
  return set(nouns),set(verbs)
      
def main():
  args = parseParams()
  nouns, verbs = getvocab(args)
  print(len(nouns))
  print(nouns)
  
  with open('train.idxs') as f:
    tidx = set([int(x) for x in f.readlines()])
  with open('nv.all') as f:
    nv = [x for i,x in enumerate(f.readlines()) if i in tidx]
  data = []
  for l in nv:
    tmp = []
    for x in l.lower().split(" "):
      w,cat = x.split("_")
      if cat[0]=="n":
        if w in nouns:
          tmp.append(w+"_N")
      elif cat[0]=="v":
        if w in verbs:
          tmp.append(w+"_V")
    if not tmp:
      tmp = ["<NO_ITEMS>"]
    data.append(" ".join(tmp))
  with open(args.out,'w') as f:
    f.write("\n".join(data))

if __name__=="__main__":
  main()
