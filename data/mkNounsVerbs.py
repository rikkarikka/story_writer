import sys


vs = []
ns = []
i = 0
with open("nv.all") as f:
    for l in f:
        print(i);i+=1
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

with open("nouns.all",'w') as f:
    f.write("\n".join(ns))
with open("verbs.all",'w') as f:
    f.write("\n".join(vs))

                
