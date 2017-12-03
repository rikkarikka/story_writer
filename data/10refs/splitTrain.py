from random import sample
with open("train.idxs") as f:
  tidx = f.read().strip().split('\n')

vidx = sample(tidx,500)
with open("t_val.idxs",'w') as f:
  f.write("\n".join(vidx))

ttidx = [x for x in tidx if x not in vidx]

with open("t_train.idxs",'w') as f:
  f.write("\n".join(ttidx))
