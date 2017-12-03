with open("t_train.idxs") as f:
  tidxs = [int(x) for x in f.read().strip().split("\n")]
with open("t_val.idxs") as f:
  vidxs = [int(x) for x in f.read().strip().split("\n")]

with open("titles.all.ner") as f:
  titles = f.read().strip().split('\n')
with open('stories.all.ner') as f:
  stories = f.read().strip().split('\n')
with open('../nv.all') as f:
  nv = f.read().strip().split('\n')

t1 = open("t_seq1.txt",'w')
t2 = open("t_seq2.txt",'w')
t1nb = open("t_seq1_nobe.txt",'w')
t2nb = open("t_seq2_nobe.txt",'w')
for k in tidxs:
  nvp = [y.split("_")[0].lower() for y in nv[k].split(" ")]
  nvp = ' '.join(nvp)
  t1.write(titles[k]+"\t"+nvp+'\n')
  t2.write(nvp+'\t'+stories[k]+'\n')
  nob = [x for x in nvp.split(" ") if x not in "is am are were was be being been".split()]
  nob = " ".join(nob)
  t1nb.write(titles[k]+"\t"+nob+'\n')
  t2nb.write(nob+'\t'+stories[k]+'\n')

  
t1 = open("v_seq1.txt",'w')
t2 = open("v_seq2.txt",'w')
t1nb = open("v_seq1_nobe.txt",'w')
t2nb = open("v_seq2_nobe.txt",'w')
for k in vidxs:
  nvp = [y.split("_")[0].lower() for y in nv[k].split(" ")]
  nvp = ' '.join(nvp)
  t1.write(titles[k]+"\t"+nvp+'\n')
  t2.write(nvp+'\t'+stories[k]+'\n')
  nob = [x for x in nvp.split(" ") if x not in "is am are were was be being been".split()]
  nob = " ".join(nob)
  t1nb.write(titles[k]+"\t"+nob+'\n')
  t2nb.write(nob+'\t'+stories[k]+'\n')



