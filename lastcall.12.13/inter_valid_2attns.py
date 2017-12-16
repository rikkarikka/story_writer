import sys
import os
import argparse
import torch
from itertools import product
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from hier_preprocess import load_data
from arguments import s2s_hier_cats as parseParams
from inter_cats_2attns import model

def validate(M,DS,args):
  print(args.valid)
  data = DS.new_data(args.valid)
  cc = SmoothingFunction()
  M.eval()
  refs = []
  hyps = []
  for sources,targets in data:
    sources = Variable(sources.cuda(),volatile=True)
    M.zero_grad()
    #logits = M(sources,None)
    #logits = torch.max(logits.data.cpu(),2)[1]
    #logits = [list(x) for x in logits]
    logits = M.beamsearch(sources)
    hyp = [DS.vocab[x] for x in logits]
    hyps.append(hyp)
    refs.append(targets)
  bleu = corpus_bleu(refs,hyps,emulate_multibleu=True,smoothing_function=cc.method3)
  M.train()
  with open(args.savestr+"hyps"+args.epoch,'w') as f:
    hyps = [' '.join(x) for x in hyps]
    f.write('\n'.join(hyps))
  try:
    os.stat(args.savestr+"refs")
  except:
    with open(args.savestr+"refs",'w') as f:
      refstr = []
      for r in refs:
        r = [' '.join(x) for x in r]
        refstr.append('\n'.join(r))
      f.write('\n'.join(refstr))
  return bleu

class jointloss:
  def __init__(self,args,optimizer):
    weights = torch.cuda.FloatTensor(args.vsz).fill_(1)
    weights[0] = 0
    self.criterion0 = nn.CrossEntropyLoss(weights)
    weights = torch.cuda.FloatTensor(args.vvsz).fill_(1)
    weights[0] = 0
    self.criterion1 = nn.CrossEntropyLoss(weights)
    weights = torch.cuda.FloatTensor(args.nvsz).fill_(1)
    weights[0] = 0
    self.criterion2 = nn.CrossEntropyLoss(weights)
    self.optimizer = optimizer

  def vloss(self,surface,logits):
    size = min(logits.size(1),surface.size(1))
    logits = logits[:,:size,:].contiguous()
    surface = surface[:,:size].contiguous()
    logits = logits.view(-1,logits.size(2))
    surface = surface.view(-1)
    loss = self.criterion0(logits,surface)
    return loss

  def calc(self,surface,nouns,verbs,logits,nouts,vouts):
    nouts = nouts[:,:nouts.size(1)-1,:].contiguous()
    vouts = vouts[:,:vouts.size(1)-1,:].contiguous()
    verbs = verbs[:,1:].contiguous()
    nouns = nouns[:,1:].contiguous()
    nouts = nouts.view(-1,nouts.size(2))
    nouns = nouns.view(-1)
    vouts = vouts.view(-1,vouts.size(2))
    verbs = verbs.view(-1)
    logits = logits.view(-1,logits.size(2))
    surface = surface.view(-1)
    loss = self.criterion0(logits,surface) + self.criterion1(vouts,verbs) + self.criterion2(nouts,nouns)
    return loss

def train(M,DS,args,jl):
  trainloss = []
  while True:
    x = DS.get_batch()
    if not x:
      break
    sources,targets,verbs,nouns = x
    sources = Variable(sources.cuda())
    targets = Variable(targets.cuda())
    verbs = Variable(verbs.cuda())
    nouns = Variable(nouns.cuda())

    M.zero_grad()
    logits,nout,vout = M(sources,verbs,nouns,targets)

    loss = jl.calc(targets,nouns,verbs,logits,nout,vout)
    loss.backward()
    jl.optimizer.step()
    loss = loss.data.cpu()[0]
    trainloss.append(loss)

    if len(trainloss)%100==99: print(trainloss[-1])
  return sum(trainloss)/len(trainloss)

def valid(M,DS,args,jl):
  print(args.valid)
  data = DS.new_data(args.valid)
  cc = SmoothingFunction()
  M.eval()
  refs = []
  hyps = []
  output = args.savestr+".outputs"
  f = open(output,'w')
  for sources,targets in data:
    sources = Variable(sources.cuda(),volatile=True)
    M.zero_grad()

    M.zero_grad()
    logits,nout,vout = M.validate(sources)
    _, preds = torch.max(logits,2)
    for x in preds:
      out = ' '.join([DS.vocab[y.cpu().data[0]] for y in x])
      if "<eos>" in out:
        out = out.split("<eos>")[0]
      f.write(out+'\n')
      print(out)



def main(args):
  DS = torch.load(args.datafile)
  if args.debug:
    args.bsz=2
    DS.train = DS.train[:2]

  args.vsz = DS.vsz
  args.svsz = DS.svsz
  args.vvsz = DS.vvsz
  args.nvsz = DS.nvsz
  if args.resume:
    M,jl = torch.load(args.resume)
    M.enc.flatten_parameters()
    M.inter.flatten_parameters()
    M.dec.flatten_parameters()
  else:
    M = model(args).cuda()
    optimizer = torch.optim.Adam(M.parameters(), lr=args.lr)
    jl = jointloss(args,optimizer)
    e=0
  M.endtok = DS.vocab.index("<eos>")
  M.punct = [DS.vocab.index(t) for t in ['.','!','?'] if t in DS.vocab]
  print(M)
  print(args.datafile)
  print(args.savestr)
  valid(M,DS,args,jl)

if __name__=="__main__":
  args = parseParams()
  if "bland" in args.savestr:
    print("set save dir to hierarchical")
    exit()
  main(args)
