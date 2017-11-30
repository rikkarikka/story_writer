import sys
import os
import argparse
import torch
from itertools import product
from torch import nn
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from arguments import s2s2s_eval as parseParams
#s2ds = torch.load("data/seq2seq2seq/s2.pt")
#vocab = s2ds.vocab

def dos1(M,DS,args,vocab):
  data = DS.val_batches
  M.eval()
  refs = []
  hyps = []
  for x in data:
    sources, targets, mask = DS.vec_batch(x,targ=False)
    sources = Variable(sources.cuda(),volatile=True)
    M.zero_grad()
    logits = M(sources,mask,None)
    logits = torch.max(logits.data.cpu(),2)[1]
    logits = [list(x) for x in logits]
    hyp = [x[:x.index(1)] if 1 in x else x for x in logits]
    hyp = [[vocab[x] for x in y] for y in hyp]
    hyps.extend(hyp)
    refs.extend(targets)
  with open(args.savestr+"/hyps_s1_val",'w') as f:
    hypstr = [' '.join(x) for x in hyps]
    f.write('\n'.join(hypstr))
  try:
    os.stat(args.savestr+"/refs_s1_val")
  except:
    with open(args.savestr+"/refs_s1_val",'w') as f:
      refstr = []
      for r in refs:
        r = [' '.join(x) for x in r]
        refstr.append('\n'.join(r))
      f.write('\n'.join(refstr))
  return hyps


def dos2(M,DS,args,s1out):
  data = DS.val_batches
  cc = SmoothingFunction()
  M.eval()
  refs = []
  hyps = []
  ctr = 0
  for x in data:
    s,t = x
    x = (s1out[ctr:ctr+len(s)],t)
    ctr+= len(s)
    sources, targets,mask = DS.pad_batch(x,targ=False)
    sources = Variable(sources.cuda(),volatile=True)
    M.zero_grad()
    logits = M(sources,mask,None)
    logits = torch.max(logits.data.cpu(),2)[1]
    logits = [list(x) for x in logits]
    hyp = [x[:x.index(1)+1] if 1 in x else x for x in logits]
    hyp = [[DS.vocab[x] for x in y] for y in hyp]
    hyps.extend(hyp)
    refs.extend(targets)
  bleu = corpus_bleu(refs,hyps,emulate_multibleu=True,smoothing_function=cc.method3)
  M.train()
  with open(args.savestr+"/hyps_s2_val",'w') as f:
    hyps = [' '.join(x) for x in hyps]
    f.write('\n'.join(hyps))
  try:
    os.stat(args.savestr+"/refs_s2_val")
  except:
    with open(args.savestr+"/refs_s2_val",'w') as f:
      refstr = []
      for r in refs:
        r = [' '.join(x) for x in r]
        refstr.append('\n'.join(r))
      f.write('\n'.join(refstr))
  return bleu

if __name__=="__main__":
  args = parseParams()
  from preprocess import load_data,vecs
  DS = torch.load(args.data)
  DS.mkbatches(args.bsz)
  from s1_preprocess import load_data,vecs
  s1ds = torch.load(args.s1ds)
  vocab = s1ds.vocab
  print(vocab)
  from s2_preprocess import load_data,vecs
  s2ds = torch.load(args.s2ds)
  DS.vocab = s2ds.vocab
  DS.stoi = s2ds.stoi

  from s2s1 import model
  m1,_ = torch.load(args.s1)
  m1.enc.flatten_parameters()
  m1.dec.flatten_parameters()
  from s2s2 import model
  m2,_ = torch.load(args.s2)
  m2.enc.flatten_parameters()
  m2.dec.flatten_parameters()
  
  s1out = dos1(m1,DS,args,vocab)
  bleu = dos2(m2,DS,args,s1out)
  print(bleu)
