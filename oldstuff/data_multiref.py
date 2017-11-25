import sys
import torch
import torchtext
from itertools import product
from torch import nn
from torch.autograd import Variable


class load_data:
  def __init__(self,bsz,train="data/train.txt.ner",valid="data/valid.txt.ner"):
    train = self.ds(train)
    val = self.ds(valid)
    

    train_sources,itos,stoi = self.numberize(train[0])
    val_sources, self.itos_sources, self.stoi_sources = self.numberize(val[0],(itos,stoi))
    train_targets,itos,stoi = self.numberize(train[1])
    val_targets, self.itos_targets, self.stoi_targets= self.numberize(val[1],(itos,stoi),multi=True)

    train_data = list(zip(train_sources,train_targets))
    train_data.sort(key=lambda x: len(x[0]),reverse=True)
    val_data = list(zip(val_sources,val_targets))
    val_data.sort(key=lambda x: len(x[0]),reverse=True)
    self.bsz = bsz

    self.gl = torchtext.vocab.GloVe(name='6B', dim=300, unk_init=torch.FloatTensor.uniform_)
    self.vecs = self.matrix(self.itos_sources) # torch.stack([self.gl[x].squeeze() for x in self.itos_sources])
    self.vecs[0].zero_()

    self.train_data = train_data
    self.titles = [x[0] for x in self.train_data]
    self.mkneighbors(self.titles)


    self.train_batches = self.batch(train_data)
    self.val_batches = self.batch(val_data,multi=True)
     
  def nn(self,title,k=1):
    title = self.numberize([title],v=(self.itos_sources,self.stoi_sources))[0][0]
    tmp = []
    for w in title:
      w = self.vecs[w]
      w = w/w.mean()
      tmp.append(w.squeeze())
    title = torch.stack(tmp).sum(0)#torch.stack([self.gl[self.itos_sources[x]].squeeze() for x in title]).sum(0)
    title = title/title.mean()
    title = title.unsqueeze(1)
    vals = torch.mm(self.neighbors,title)
    _,best = torch.sort(vals,0,True)
    return best[:k]

  def lookup_title(self,idx):
    closest_title = ' '.join([x.itos_sources[y] for y in self.titles[k[0]]])
    return closest_title
    
  def mkneighbors(self,titles):
    vecs = []
    for t in titles:
      tmp = torch.stack([self.gl[self.itos_sources[x]].squeeze() for x in t]).sum(0)
      tmp = tmp/tmp.mean()
      vecs.append(tmp)
    self.neighbors = torch.stack(vecs)
    
  def matrix(self,li):
    return torch.stack([self.gl[x]/self.gl[x].mean() for x in li])

  def batch(self,data,multi=False):
    ctr = 0
    batches = []
    while ctr < len(data):
      tmp = data[ctr:ctr+self.bsz]
      ctr+=self.bsz
      msour = max([len(x[0]) for x in tmp])
      mtarg = max([len(x[1]) for x in tmp])
      source = torch.LongTensor(self.bsz,msour).zero_()
      if multi:
        target = [d[1] for d in tmp]
      else:
          target = torch.LongTensor(self.bsz,mtarg).zero_()
      for i,d in enumerate(tmp):
        for j,w in enumerate(d[0]): source[i,j] = w
        if not multi:
            for j,w in enumerate(d[1]): target[i,j] = w
      batches.append((self.to_vecs(source),target,source))
    return batches


  def numberize(self,data,v=None,multi=False):
    if v:
      itos, stoi = v
    else:
      itos = ['<pad>']
      stoi = {'<pad>':0}
    nums = []
    for k in data:
      if multi==False:
        k = [k]
      else:
        k = k.split("\t")
      out = []
      for d in k:
          #d = "<start> "+d+" <end>"
          d = d.split(" ")
          tmp = []
          for w in d:
            if w not in stoi:
              itos.append(w)
              stoi[w] = len(itos)-1
            tmp.append(stoi[w])
          out.append(tmp)
      if multi==False:
          out = out[0]
      nums.append(out)
    return nums, itos, stoi

  def ds(self,fn):
    with open(fn) as f:
      sources, targets = zip(*[x.strip().split("\t",maxsplit=1) for x in f.readlines()])
    return sources, targets

  def to_vecs(self,batch):
    v = torch.FloatTensor(batch.size(0),batch.size(1),300)
    for i,j in product(range(batch.size(0)),range(batch.size(1))):
      v[i,j,:] = self.vecs[batch[i,j]]

    return v

if __name__=="__main__":
    x = load_data(32)
    while True:
      title = input("Title: ")
      top5 = x.nn(title,5)
      for k in top5:
        print(x.lookup_title(k))
