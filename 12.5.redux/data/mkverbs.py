with open('train.idxs') as f:
  tidx = set([int(x) for x in f.readlines()])

with open('nv.all') as f:
  nv = [x for i,x in enumerate(f.readlines()) if i in tidx]

cutverbs = "was had	went got decided were did wanted".split()
print(cutverbs)
verbs = []
for x in nv:
  tmp = [k.split("_")[0] for k in x.split(" ") if "_V" in k]
  tmp = [x for x in tmp if x not in cutverbs]
  if not tmp:
    tmp = ["<NO_VERBS>"]
  verbs.append(" ".join(tmp))
with open("train.verbs",'w') as f:
  f.write("\n".join(verbs))
