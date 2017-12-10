import sys
import torch
from collections import Counter
from arguments import s2s_hier_cats as parseParams

class load_data:
  def __init__(self,args):
    self.args = args
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
    self.mkverbs(args.interfile)
    print("verb data: ",len(self.verb_data))

    self.train = list(zip(train_sources,train_targets,self.verb_data,self.noun_data))

    self.train.sort(key=lambda x: len(x[0]),reverse=True)
    self.bctr = 0
    self.bsz = args.bsz
    self.dsz = len(self.train)

  def mkverbs(self,fn):
    with open(fn) as f:
      data = f.read().strip()
    self.verb_data = []
    self.noun_data = []
    for x in data.split("\n"):
      #vtoks = [y for y in x.split(" ") if "<" not in y]
      x = x.split(" ")
      self.noun_data.append(x[::2])
      self.verb_data.append(x[1::2])
    vctr = list(set([x for y in self.verb_data for x in y]))
    nctr = list(set([x for y in self.noun_data for x in y]))
    self.verb_vocab = ["<pad>","<eos>","<unk>","<start>"]+vctr
    self.noun_vocab = ["<pad>","<eos>","<unk>","<start>"]+nctr
    self.noun_data = [[3]+[self.noun_vocab.index(x) if x in self.noun_vocab else 2 for x in y]+[1] for y in self.noun_data]
    self.verb_data = [[3]+[self.verb_vocab.index(x) if x in self.verb_vocab else 2 for x in y]+[1] for y in self.verb_data]
    print("verb vocab size: ",len(self.verb_vocab))
    print("noun vocab size: ",len(self.noun_vocab))
    self.vvsz = len(self.verb_vocab)
    self.nvsz = len(self.noun_vocab)
  
  def get_batch(self):
    if self.bctr>=self.dsz:
      self.bctr = 0
      return None
    else:
      data = self.train
      siz = len(data[self.bctr][0])
      k = 0
      srcs,tgts,verbs,nouns = [],[],[],[]
      while k<self.bsz and self.bctr+k<self.dsz:
        src,tgt,verb,noun = data[self.bctr+k]
        if len(src)<siz:
          break
        srcs.append(src)
        tgts.append(tgt)
        verbs.append(verb)
        nouns.append(noun)
        k+=1
      self.bctr+=k
    return self.pad_batch((srcs,tgts,verbs,nouns))

  def new_data(self,fn):
    src,tgt = self.ds(fn)
    new = []
    for i in range(len(src)):
      new.append(self.pad_batch(([src[i]],tgt[i]),targ=False,v=False))
    return new

  def pad_batch(self,batch,targ=True,v=True):
    if v:
      srcs,tgts,verbs,nouns = batch
    else:
      srcs,tgts = batch
    targs = tgts
    srcnums = [[self.stoi[w] if w in self.stoi else 2 for w in x]+[1] for x in srcs]
    m = max([len(x) for x in srcnums])
    srcnums = [x+([0]*(m-len(x))) for x in srcnums]
    if self.args.cuda:
      tensor = torch.cuda.LongTensor(srcnums)
    else:
      tensor = torch.LongTensor(srcnums)
    if targ:
      targtmp = [[self.vocab.index(w) if w in self.vocab else 2 for w in x]+[1] for x in tgts]
      m = max([len(x) for x in targtmp])
      targtmp = [x+([0]*(m-len(x))) for x in targtmp]
      if self.args.cuda:
        targs = torch.cuda.LongTensor(targtmp)
      else:
        targs = torch.LongTensor(targtmp)
    if v:
      m = max([len(x) for x in verbs])
      targtmp = [x+([0]*(m-len(x))) for x in verbs]
      if self.args.cuda:
        vtensor = torch.cuda.LongTensor(targtmp)
      else:
        vtensor = torch.LongTensor(targtmp)
      m = max([len(x) for x in nouns])
      targtmp = [x+([0]*(m-len(x))) for x in nouns]
      if self.args.cuda:
        ntensor = torch.cuda.LongTensor(targtmp)
      else:
        ntensor = torch.LongTensor(targtmp)
      
    if v:
      return (tensor,targs,vtensor,ntensor)
    else:
      return (tensor,targs)

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
