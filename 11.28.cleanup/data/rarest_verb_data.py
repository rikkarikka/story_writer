import sys
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer as WNL


def load_verbs():
  with open("verb_cats.txt") as f:
    cats = ['<no_verb>']+f.read().strip().split('\n')
  wordassoc = {}
  with open("verb_vocab.txt") as f:
    ctr = 1
    for l in f:
      l = l.strip()
      if l == "___":
        ctr+=1
      else:
        if l not in wordassoc:
          if l not in ['was','is','be','been','being','get','got','were','had','has','have','are','will']:
            wordassoc[l] = ctr
  return cats,wordassoc

try:
  with open("vdict.pickle",'rb') as f:
    vcats,vdict = pickle.load(f)
except:
  vcats,vdict = load_verbs()
  with open("vdict.pickle",'wb') as f:
    pickle.dump((vcats,vdict),f)
wnl = WNL()

def getverbs(story):
  ws = story.strip().split(" ")
  lems = [wnl.lemmatize(x,pos='v') for x in ws]
  verbs = [vcats[vdict[x]] for x in lems if x in vdict]
  return verbs

def main():
  allverbs = []
  with open("train.txt.ner") as f:
    for l in f.readlines()[:2000]:
      title, story = l.split("\t")
      verbs = getverbs(story)
      allverbs.extend(verbs)
  c = Counter(allverbs)
  hist = Counter([x[1] for x in c.most_common()])
  for x in hist.most_common():
    print(x)

main()
