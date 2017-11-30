import sys


vs = []
ns = []
with open("train.idxs") as f:
  idxs = f.read().strip().split("\n")
  idxs = [int(x) for x in idxs]
i = 0
with open("nv.all") as f:
  for l in f:
    if i in idxs:
        l = l.strip().split()
        n,v = [],[]
        for w in l:
            w,pos = w.split("_")
            if pos[0] == "N":
                n.append(w)
            elif pos[0] == "V":
                v.append(w)
        ns.append(" ".join(n))
        vs.append(" ".join(v))
    i += 1
    print(i)

with open("nouns.train",'w') as f:
    f.write("\n".join(ns))
with open("verbs.train",'w') as f:
    f.write("\n".join(vs))

                
