import sys

try:
  k = int(sys.argv[1])
except:
  print('give k as arg');exit()

with open("t_tite_train.txt") as f:
  titles = f.read().strip().split('\n')
with open("t_stor_train.txt") as f:
  stories =  f.read().strip().split('\n')
with open("t_train_rare.cats") as f:
  cats =  f.read().strip().split('\n')

f = open("s1rarecats.train",'w')
g = open("s2rarecats.train",'w')
for i in range(len(titles)):
  n,v = cats[i].split(";")
  n = " ".join(n.strip().split(" ")[:k])
  v = " ".join(v.strip().split(" ")[:k])
  f.write(titles[i]+'\t'+n+" ; "+v+'\n')
  g.write(n+" ; "+v+'\t'+stories[i]+'\n')

f.close()
g.close()
with open("t_tite_val.txt") as f:
  titles = f.read().strip().split('\n')
with open("t_stor_val.txt") as f:
  stories =  f.read().strip().split('\n')
with open("t_val_rare.cats") as f:
  cats =  f.read().strip().split('\n')
f = open("s1rarecats.val",'w')
g = open("s2rarecats.val",'w')
for i in range(len(titles)):
  n,v = cats[i].split(";")
  n = " ".join(n.strip().split(" ")[:k])
  v = " ".join(v.strip().split(" ")[:k])
  f.write(titles[i]+'\t'+n+" ; "+v+'\n')
  g.write(n+" ; "+v+'\t'+stories[i]+'\n')
