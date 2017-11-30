import sys
import torch
import torchtext
from collections import Counter
from arguments import cat_s2s as parseParams
from collections import defaultdict
from nltk.stem import WordNetLemmatizer as WNL

class load_data:
  def __init__(self,args,train="data/train.txt.ner",valid="data/valid.txt.ner"):
    # load data
    train_sources,train_targets = self.ds(train)
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

    self.vcats, self.vvocab = self.load_verbs()
    self.train = list(zip(train_sources,train_targets))
    self.train.sort(key=lambda x: len(x[0]),reverse=True)
    val_sources, val_targets = self.ds(valid)
    self.val = list(zip(val_sources,val_targets))
    self.val.sort(key=lambda x:len(x[0]),reverse=True)
    self.mkbatches(args.bsz)

  def load_verbs(self):
    with open("data/verb_cats.txt") as f:
      cats = f.read().strip().split('\n')
    wordassoc = {}
    with open("data/verb_vocab.txt") as f:
      ctr = 0
      for l in f:
        l = l.strip()
        if l == "___":
          ctr+=1
        else:
          if l not in wordassoc:
            if l not in ['was','is','be','been','being','get','got','were','had','has','have','are','will']:
              wordassoc[l] = ctr
    return cats,wordassoc


  def pad_batch(self,batch,targ=True,align=False):
    srcs,tgts,valign = batch
    vmatrix = None
    targs = tgts
    srcnums = [[self.stoi[w] if w in self.stoi else 2 for w in x]+[1] for x in srcs]
    tensor = torch.cuda.LongTensor(srcnums)
    if targ:
      targtmp = [[self.vocab.index(w) if w in self.vocab else 2 for w in x]+[1] for x in tgts]
      m = max([len(x) for x in targtmp])
      targtmp = [x+([0]*(m-len(x))) for x in targtmp]
      targs = torch.cuda.LongTensor(targtmp)
      if align:
        vmatrix = torch.cuda.FloatTensor(len(tgts),m,len(self.vcats)).zero_()
        for b in range(len(valign)):
          for i in range(len(valign[b])):
            #if valign[b][i] == 0:
            #  vmatrix[b,i,:].fill_(1/len(self.vcats))
            #else:
            if valign[b][i]>0:
              vmatrix[b,i,valign[b][i]-1]=1
              
    return (tensor,targs,vmatrix)

  def mkbatches(self,bsz):
    self.bsz = bsz
    self.train_batches = self.batches(self.train,align=True)
    self.val_batches = self.batches(self.val)

  def batches(self,data,align=False):
    wnl = WNL()
    ctr = 0
    batches = []
    while ctr<len(data):
      siz = len(data[ctr][0])
      k = 0
      srcs,tgts = [],[]
      alignments = []
      while k<self.bsz and ctr+k<len(data):
        src,tgt = data[ctr+k]
        if len(src)<siz:
          break
        if align:
          lems = [wnl.lemmatize(x,pos='v') for x in tgt]
          valign = [self.vvocab[x]+1 if x in self.vvocab else 0 for x in tgt]
          alignments.append(valign)
        srcs.append(src)
        tgts.append(tgt)
        k+=1
      ctr+=k
      batches.append((srcs,tgts,alignments))
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
  print(DS.train_batches[0])
  torch.save(DS,args.datafile)
