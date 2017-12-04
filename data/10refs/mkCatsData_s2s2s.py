with open('t_tite_train.txt') as f:
  tites = f.read().strip().split('\n')
with open('t_stor_train.txt') as f:
  stors= f.read().strip().split('\n')
with open('t_train.cats') as f:
  cats = f.read().strip().split('\n')

one = open('s2cats1.train','w')
two = open('s2cats2.train','w')
for i in range(len(cats)):
  if cats[i]:
    one.write(tites[i]+"\t"+cats[i]+'\n')
    two.write(cats[i]+"\t"+stors[i]+'\n')

with open('t_tite_val.txt') as f:
  tites = f.read().strip().split('\n')
with open('t_stor_val.txt') as f:
  stors= f.read().strip().split('\n')
with open('t_val.cats') as f:
  cats = f.read().strip().split('\n')

one = open('s2cats1.val','w')
two = open('s2cats2.val','w')
for i in range(len(cats)):
  if cats[i]:
    one.write(tites[i]+"\t"+cats[i]+'\n')
    two.write(cats[i]+"\t"+stors[i]+'\n')
