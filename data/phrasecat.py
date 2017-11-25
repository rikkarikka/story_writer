import sys

# replaces phrases from vocabulary.txt in data

with open("vocabulary.txt") as f:
  phrases = [x.strip() for x in f.readlines() if "_" in x]

for fn in ['train.txt.ner','valid.txt.ner','test.txt.ner']:
  with open(fn) as f:
    data = f.read()
  for p in phrases:
    p = p.replace(" ","_")
    pprime = p.replace("_"," ")
    data = data.replace(pprime, p)
  with open(fn+".phrases",'w') as f:
    f.write(data)

