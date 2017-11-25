import sys
import torch
import torchtext
from collections import Counter

class vecs:
  def __init__(self):
    
    self.gl = torchtext.vocab.GloVe(name='6B', dim=300, unk_init=torch.FloatTensor.uniform_)
    self.cache = {}

  def get(self,w):
    if w not in self.cache:
      tmp = self.gl[w]
      tmp = tmp/tmp.norm()
      self.cache[w] = tmp.squeeze()
    return self.cache[w]

class load_data:
  def __init__(self,train="data/train.txt.ner",valid="data/valid.txt.ner"):
    train_sources,train_targets = self.ds(train)
    train_targets = [x[0] for x in train_targets]
    ctr = Counter([x for z in train_targets for x in z])
    thresh = 3
    self.vocab = ["<pad>","<eos>","<unk>","<start>"]+[x for x in ctr if ctr[x]>thresh]
    self.vsz = len(self.vocab)
    self.train = list(zip(train_sources,train_targets))
    self.train.sort(key=lambda x: len(x[0]),reverse=True)
    val_sources, val_targets = self.ds(valid)
    self.val = list(zip(val_sources,val_targets))
    self.val.sort(key=lambda x:len(x[0]),reverse=True)
    self.vecs = vecs()
    self.train = self.train[:2]
    self.val = self.val[:2]

  def mkbatches(self,bsz):
    self.bsz = bsz
    self.train_batches = self.batches(self.train)
    self.val_batches = self.batches(self.val)

  def vec_batch(self,batch,targ=True):
    srcs,tgts = batch
    targs = tgts
    tensor = torch.cuda.FloatTensor(len(srcs),max([len(x) for x in srcs]),300).zero_()
    mask = torch.cuda.ByteTensor(tensor.size(0),tensor.size(1)).fill_(1)
    for i,s in enumerate(srcs):
      for j, w in enumerate(s):
        mask[i,j]=0
        tensor[i,j,:] = self.vecs.get(w)
    if targ:
      targtmp = [[self.vocab.index(w) if w in self.vocab else 2 for w in x]+[1] for x in tgts]
      m = max([len(x) for x in targtmp])
      targtmp = [x+([0]*(m-len(x))) for x in targtmp]
      targs = torch.cuda.LongTensor(targtmp)
    return (tensor,targs,mask)
    
  def batches(self,data):
    ctr = 0
    batches = []
    while ctr<len(data):
      srcs,tgts= zip(*data[ctr:ctr+self.bsz])
      ctr += self.bsz
      batches.append((srcs,tgts))
    return batches
        
  def ds(self,fn):
    with open(fn) as f:
      sources, targs = zip(*[x.strip().split("\t",maxsplit=1) for x in f.readlines()])
    sources = [x.split(" ") for x in sources]
    targets = []
    for t in targs:
      t = t.replace("PERSON","<person>").replace("LOCATION","<location>").lower()
      t = t.split('\t')
      tmp = []
      for x in t:
        tmp.append(x.split(" "))
      targets.append(tmp)
    return sources, targets

if __name__=="__main__":
  DS = load_data()
  torch.save(DS,"data/multiref_debug.pt")
