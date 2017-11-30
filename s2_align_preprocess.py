import sys
import pickle
import torch
import torchtext
from collections import Counter
from arguments import s2s2_params as parseParams
from nltk.stem import WordNetLemmatizer as WNL

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

class aligner:
  def __init__(self,cats,vocab,ndic):

    with open(ndic,'rb') as f:
      self.nouns = pickle.load(f)

    with open(cats) as f:
      self.cats = f.read().strip().split("\n")
    self.wnl = WNL()

    wordassoc = {}
    with open(vocab) as f:
      ctr = 0
      for l in f:
        l = l.strip()
        if l == "___":
          ctr+=1
        else:
          if l not in wordassoc:
            if l not in ['was','is','be','been','being','get','got','were','had','has','have','are','will']:
              wordassoc[l] = ctr
    self.wordassoc = wordassoc

  def align(self,src,tgt):
    avec = []
    for w in tgt:
      l = wnl.lemmatize(w,pos='v') 
      c = self.wordassoc[l] if l in self.wordassoc else None
      if c and c in src:
        n = src.index(c)
      elif w 
      else:
        n = -1
      avec.append(n)

      
      

class load_data:
  def __init__(self,args,train="data/train.txt.ner",valid="data/valid.txt.ner"):
    self.args = args

    train_sources,train_targets = self.ds(train)
    train_targets = [x[0] for x in train_targets]
    ctr = Counter([x for z in train_targets for x in z])
    thresh = 3
    self.vocab = ["<pad>","<eos>","<unk>","<start>"]+[x for x in ctr if ctr[x]>thresh]
    ctr = Counter([x for z in train_sources for x in z])
    thresh = 3
    self.itos = ["<pad>","<eos>","<unk>","<start>"]+[x for x in ctr if ctr[x]>thresh]
    self.stoi = {x:i for i,x in enumerate(self.itos)}
    self.vsz = len(self.vocab)
    self.train = list(zip(train_sources,train_targets))
    self.train.sort(key=lambda x: len(x[0]),reverse=True)

    align = aligner()
    self.trainalign = [align.align[x] for x in self.train]

    if not args.pretrain:
      val_sources, val_targets = self.ds(valid)
      self.val = list(zip(val_sources,val_targets))
      self.val.sort(key=lambda x:len(x[0]),reverse=True)

    self.vecs = vecs()
    self.events = self.init_vocab(args)


  def init_vocab(self,args):
    #get a vsz x dim matrix to give to the model
    with open(args.eventvocab) as f:
      vocab = f.read().strip().lower().split("\n")
    matrix = self.matrix(vocab).cuda()
    return matrix

  def matrix(self,li):
    return torch.stack([self.vecs.get(x) for x in li])

  def mkbatches(self,bsz,align=False):
    self.bsz = bsz
    if align:
      self.train_batches,self.align_batches = self.batches(self.train,align=self.trainalign)
    else:
      self.train_batches = self.batches(self.train)
    self.val_batches = self.batches(self.val)

  def pad_batch(self,batch,targ=True):
    srcs,tgts = batch
    targs = tgts
    tensor = torch.cuda.LongTensor(len(srcs),max([len(x) for x in srcs])).zero_()
    mask = torch.cuda.ByteTensor(tensor.size(0),tensor.size(1)).fill_(1)
    for i,s in enumerate(srcs):
      for j, w in enumerate(s):
        mask[i,j]=0
        tensor[i,j] = self.stoi[w] if w in self.stoi else 2
    if targ:
      targtmp = [[self.vocab.index(w) if w in self.vocab else 2 for w in x]+[1] for x in tgts]
      m = max([len(x) for x in targtmp])
      targtmp = [x+([0]*(m-len(x))) for x in targtmp]
      targs = torch.cuda.LongTensor(targtmp)
    return (tensor,targs,mask)
    
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
    
  def batches(self,data,align=None):
    ctr = 0
    batches = []
    ab = []
    while ctr<len(data):
      srcs,tgts= zip(*data[ctr:ctr+self.bsz])
      if align:
        ab.append(align[ctr:ctr+self.bsz])
      ctr += self.bsz
      batches.append((srcs,tgts))
    if align:
      return batches,ab
    else:
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
  if args.pretrain:
    print('pretrain')
    DS = load_data(args)
    torch.save(DS,"data/pretrain.pt")
  else:
    DS = load_data(args,train=args.train,valid=args.valid)
    torch.save(DS,args.save)
