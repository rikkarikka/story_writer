import sys
from collections import defaultdict
from nltk.stem import WordNetLemmatizer as WNL

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
        if l not in ['was','is','be','been','being','get','got','were','had','has','have','are','will']:
          wordassoc[l] = ctr

print(len(wordassoc))
used = []
new = open("verbs2cats.train",'w')
with open("verbs.train") as f:
  for l in f:
    txt = l.strip().split()
    lems = [wnl.lemmatize(x,pos='v') for x in txt]
    txt = [(wordassoc[x],txt[i]) for i,x in enumerate(lems) if x in wordassoc]
    tmp = []
    for cat,word in txt:
      tmp.append(word+"->"+cats[cat])
      used.append(cats[cat])
    txt = ' '.join(tmp)
    new.write(txt+'\n')
new.close()
print(len(set(used)))

