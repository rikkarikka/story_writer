import sys

with open(sys.argv[1]) as f:
  data = f.readlines()

with open(sys.argv[2],'w') as f:
  for l in data:
    l = l.lower().strip()
    if "n't" in l:
      l = l.replace("n't"," n`t")
    for x in ".,?!')(][":
      l = l.replace(x, ' '+x)
    l = l.replace("n`t","n't")
    f.write(l+"\n")
