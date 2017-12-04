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

with open("t_train.cats") as f:
  with open('t_val.cats') as g:
    cats = f.read().split() + g.read().split()

verbs = [x for x in cats if x.isdigit()]
print("processed nouns and verbs")
nouns = [x for x in cats if not x.isdigit()]
print("processed nouns and verbs")

_, catsd = make_wordd()
nc = Counter(nouns)
vc = Counter(verbs)
f = open("nounfreq.cats",'w')
g = open("verbfreq.cats",'w')
for n,c in nc.most_common():
  f.write(n+'\t'+str(c)+'\n')
for v,c in vc.most_common():
  v = catsd[int(v)]
  g.write(v+'\t'+str(c)+'\n')
  
