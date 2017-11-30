import sys

with open(sys.argv[1]) as f:
  data = f.readlines()

story_overlaps = []
allwords = []
sent_overlap =[]
allsents = []
for l in data:
  words = l.strip().split(" ")
  if words[-1] == '<eos>': words = words[:-1]
  story_overlaps.append(len(words)/len(set(words)))
  allwords.extend(words)
  sents = l.split(" . ")
  sent_overlap.append(len(sents)/len(set(sents)))
  allsents.extend(sents)



print("story overlap : ",sum(story_overlaps)/len(story_overlaps))
print("corpus overlap : ",len(allwords)/len(set(allwords)))
print("Vocab use : ",len(set(allwords)))
print("avg sent overlap per story : ",sum(sent_overlap)/len(sent_overlap))
print("corpus sent overlap : ",len(allsents)/len(set(allsents)))

