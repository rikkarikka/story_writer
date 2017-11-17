import sys
import torch
import torchtext
from itertools import product
from torch import nn
from torch.autograd import Variable


class load_data:
  def __init__(self,bsz,debug=False,train="data/train.txt.phrases",valid="data/valid.txt.phrases"):
    if debug:
      train = self.ds("two.txt")
      val = self.ds("two.txt")
    else:
      train = self.ds(train)
      val = self.ds(valid)
    

    train_sources,itos,stoi = self.numberize(train[0])
    val_sources, self.itos_sources, self.stoi_sources = self.numberize(val[0],(itos,stoi))
    train_targets,itos,stoi = self.numberize(train[1])
    val_targets, self.itos_targets, self.stoi_targets= self.numberize(val[1],(itos,stoi))

    train_data = list(zip(train_sources,train_targets))
    train_data.sort(key=lambda x: len(x[0]),reverse=True)
    val_data = list(zip(val_sources,val_targets))
    val_data.sort(key=lambda x: len(x[0]),reverse=True)
    self.bsz = bsz

    self.gl = torchtext.vocab.GloVe(name='6B', dim=300, unk_init=torch.FloatTensor.uniform_)
    self.vecs = self.matrix(self.itos_sources) # torch.stack([self.gl[x].squeeze() for x in self.itos_sources])
    self.vecs[0].zero_()

    self.train_batches = self.batch(train_data)
    self.val_batches = self.batch(val_data)
     
  def matrix(self,li):
    return torch.stack([self.gl[x].squeeze() for x in li])

  def batch(self,data):
    ctr = 0
    batches = []
    while ctr < len(data):
      tmp = data[ctr:ctr+self.bsz]
      ctr+=self.bsz
      msour = max([len(x[0]) for x in tmp])
      mtarg = max([len(x[1]) for x in tmp])
      source = torch.LongTensor(self.bsz,msour).zero_()
      target = torch.LongTensor(self.bsz,mtarg).zero_()
      for i,d in enumerate(tmp):
        for j,w in enumerate(d[0]): source[i,j] = w
        for j,w in enumerate(d[1]): target[i,j] = w
      batches.append((self.to_vecs(source),target,source))
    return batches


  def numberize(self,data,v=None):
    if v:
      itos, stoi = v
    else:
      itos = ['<pad>']
      stoi = {'<pad>':0}
    nums = []
    for d in data:
      d = "<start> "+d+" <end>"
      d = d.split(" ")
      tmp = []
      for w in d:
        if w not in stoi:
          itos.append(w)
          stoi[w] = len(itos)-1
        tmp.append(stoi[w])
      nums.append(tmp)
    return nums, itos, stoi

  def ds(self,fn):
    with open(fn) as f:
      sources, targets = zip(*[x.strip().split("\t") for x in f.readlines()])
    return sources, targets

  def to_vecs(self,batch):
    v = torch.FloatTensor(batch.size(0),batch.size(1),300)
    for i,j in product(range(batch.size(0)),range(batch.size(1))):
      v[i,j,:] = self.vecs[batch[i,j]]

    return v

