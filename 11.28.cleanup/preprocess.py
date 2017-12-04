import sys
import torch
from collections import Counter
from arguments import s2s as parseParams

class load_data:
  def __init__(self,args):
    train_sources,train_targets = self.ds(args.train)
    train_targets = [x[0] for x in train_targets]
    ctr = Counter([x for z in train_targets for x in z])
    thresh = 3
    self.vocab = ["<pad>","<eos>","<unk>","<start>"]+[x for x in ctr if ctr[x]>thresh]
    self.vsz = len(self.vocab)
    ctr = Counter([x for z in train_sources for x in z])
    thresh = 1
    self.itos = ["<pad>","<eos>","<unk>","<start>"]+[x for x in ctr if ctr[x]>thresh]
    self.stoi = {x:i for i,x in enumerate(self.itos)}
    self.svsz = len(self.itos)
    self.train = list(zip(train_sources,train_targets))
    self.train.sort(key=lambda x: len(x[0]),reverse=True)
    val_sources, val_targets = self.ds(args.valid)
    self.val = list(zip(val_sources,val_targets))
    self.val.sort(key=lambda x:len(x[0]),reverse=True)
    self.mkbatches(args.bsz)

  def new_data(self,src):
    with open(src) as f:
      new = [(x.strip().split(),self.val[i][1]) for i,x in enumerate(f.readlines())]
    new.sort(key=lambda x:len(x[0]),reverse=True)
    self.new_batches = self.batches(new)


  def pad_batch(self,batch,targ=True):
    srcs,tgts = batch
    targs = tgts
    srcnums = [[self.stoi[w] if w in self.stoi else 2 for w in x]+[1] for x in srcs]
    tensor = torch.cuda.LongTensor(srcnums)
    if targ:
      targtmp = [[self.vocab.index(w) if w in self.vocab else 2 for w in x]+[1] for x in tgts]
      m = max([len(x) for x in targtmp])
      targtmp = [x+([0]*(m-len(x))) for x in targtmp]
      targs = torch.cuda.LongTensor(targtmp)
    return (tensor,targs)

  def mkbatches(self,bsz):
    self.bsz = bsz
    self.train_batches = self.batches(self.train)
    self.val_batches = self.batches(self.val)

  def batches(self,data):
    ctr = 0
    batches = []
    while ctr<len(data):
      siz = len(data[ctr][0])
      k = 0
      srcs,tgts = [],[]
      while k<self.bsz and ctr+k<len(data):
        src,tgt = data[ctr+k]
        if len(src)<siz:
          break
        srcs.append(src)
        tgts.append(tgt)
        k+=1
      ctr+=k
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
  args = parseParams()
  DS = load_data(args)
  torch.save(DS,args.datafile)
