import sys
import torch
import torchtext
from itertools import product
from torch import nn
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

def calcbleu(gen, targets):
  #calcbleu(generated, targets, DS.stoi_targets["<end>"]):
  cc = SmoothingFunction()
  bleu = sentence_bleu(targets,gen,smoothing_function=cc.method3)
  return bleu

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
    self.train = self.ds(train)
    self.val = self.ds(valid)
    self.rawtitles = self.train[0]
    
    self.vecs = vecs()
    self.titles = self.mktitles([x for x in self.train[0]])
    
  def ds(self,fn):
    with open(fn) as f:
      sources, targs = zip(*[x.strip().split("\t",maxsplit=1) for x in f.readlines()])
    targets = []
    for t in targs:
      t = t.split('\t')
      tmp = []
      for x in t:
        tmp.append(x.split(" "))
      targets.append(tmp)
    
    return sources, targets

  def mktitles(self,data):
    titles = []
    for x in data:
      tmp = torch.stack([self.vecs.get(w) for w in x.split(" ")])
      tmp,_ = torch.max(tmp,0)
      titles.append(tmp.squeeze())
    return torch.stack(titles)

  def nn(self,title,k=1):
    v = torch.stack([self.vecs.get(w) for w in title.split(" ")])
    v,_ = torch.max(v,0)
    v = v.view(1,300)
    mul = torch.mm(self.titles,v.t())
    _,best = torch.sort(mul,0,True)
    return best[:k]

DS = load_data()
valtitles = DS.val[0]
valstories = DS.val[1]
bleu = 0
for i,title in enumerate(valtitles):
  best = DS.nn(title)[0][0]
  story = DS.train[1][best][0]
  targets = valstories[i]
  bleu += calcbleu(story,targets)
i+=1
print(bleu/i)
print(i)

