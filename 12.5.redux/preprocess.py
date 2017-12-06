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
    
    #make verbs
    self.mkverbs(args.verbs)
    print("verb data: ",len(self.verb_data))

    self.train = list(zip(train_sources,train_targets,self.verb_data))
    print(len(self.train))
    self.train.sort(key=lambda x: len(x[0]),reverse=True)
    val_sources, val_targets = self.ds(args.valid)
    self.val = list(zip(val_sources,val_targets))
    self.val.sort(key=lambda x:len(x[0]),reverse=True)
    self.mkbatches(args.bsz)

  def mkverbs(self,fn):
    with open(fn) as f:
      data = f.read().strip()
    vctr = Counter(data.split())
    self.verb_vocab = ["<pad>","<eos>","<unk>"]+[x for x in vctr if vctr[x]>1]
    print("verb vocab size: ",len(self.verb_vocab))
    self.vvsz = len(self.verb_vocab)
    self.verb_data = []
    for x in data.split("\n"):
      self.verb_data.append([self.verb_vocab.index(k) if k in self.verb_vocab else 2 for k in x.split(" ")]+[1])

  def new_data(self,src,targ=None):
    if targ is None:
      with open(src) as f:
        src,tgt = self.ds(src)
        new = list(zip(src,tgt))
        print(new)
    else:
      new = zip(src,targ)
    new.sort(key=lambda x:len(x[0]),reverse=True)
    self.new_batches = self.batches(new)


  def pad_batch(self,batch,targ=True,v=True):
    if v:
      srcs,tgts,verbs = batch
    else:
      srcs,tgts = batch
    targs = tgts
    srcnums = [[self.stoi[w] if w in self.stoi else 2 for w in x]+[1] for x in srcs]
    m = max([len(x) for x in srcnums])
    srcnums = [x+([0]*(m-len(x))) for x in srcnums]
    tensor = torch.cuda.LongTensor(srcnums)
    if targ:
      targtmp = [[self.vocab.index(w) if w in self.vocab else 2 for w in x]+[1] for x in tgts]
      m = max([len(x) for x in targtmp])
      targtmp = [x+([0]*(m-len(x))) for x in targtmp]
      targs = torch.cuda.LongTensor(targtmp)
    if v:
      m = max([len(x) for x in verbs])
      targtmp = [x+([0]*(m-len(x))) for x in verbs]
      vtensor = torch.cuda.LongTensor(targtmp)
      
    if v:
      return (tensor,targs,vtensor)
    else:
      return (tensor,targs)

  def mkbatches(self,bsz):
    self.bsz = bsz
    self.train_batches = self.batches(self.train,v=True)
    self.val_batches = self.batches(self.val)

  def batches(self,data,v=False):
    ctr = 0
    batches = []
    while ctr<len(data):
      siz = len(data[ctr][0])
      k = 0
      srcs,tgts = [],[]
      if v:
        verbbatch = []
      while k<self.bsz and ctr+k<len(data):
        if v:
          src,tgt,verbs = data[ctr+k]
        else:
          src,tgt = data[ctr+k]
        if len(src)<siz:
          break
        srcs.append(src)
        tgts.append(tgt)
        if v:
          verbbatch.append(verbs)
        k+=1
      ctr+=k
      if v:
        batches.append((srcs,tgts,verbbatch))
      else:
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
