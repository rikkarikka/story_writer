import sys

with open("vocabulary.txt") as f:
  cats = f.read().strip().split("\n")
  cats = [x.replace('_',' ') for x in cats if x!="___"]

print(len(cats))
with open("train.txt") as f:
  for l in f:
    lcats = [x for x in cats if x in l]
    cats= [x for x in cats if x not in lcats]
print(cats)
  
print(len(cats))
