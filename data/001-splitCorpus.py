import sys
import csv

titles = []
stories = []
with open(sys.argv[1]) as f:
  c = csv.reader(f)
  for x in c:
    stories.append(" ".join(x[2:]))
    titles.append(x[1])

with open("titles.all",'w') as f:
  f.write("\n".join(titles))
with open("stories.all",'w') as f:
  f.write("\n".join(stories))
