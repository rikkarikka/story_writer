import sys
import torch
import argparse
from torch.autograd import Variable
from s2s import model
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from preprocess import load_data

def validate(M,DS,savestr,data=None):
  if not data:
    data = DS.val_batches
  cc = SmoothingFunction()
  M.eval()
  refs = []
  hyps = []
  for x in data:
    sources, targets = DS.pad_batch(x,targ=False)
    sources = Variable(sources.cuda(),volatile=True)
    M.zero_grad()
    logits = M(sources,None)
    logits = torch.max(logits.data.cpu(),2)[1]
    logits = [list(x) for x in logits]
    hyp = [x[:x.index(1)+1] if 1 in x else x for x in logits]
    hyp = [[DS.vocab[x] for x in y] for y in hyp]
    hyps.extend(hyp)
    refs.extend(targets)
  bleu = corpus_bleu(refs,hyps,emulate_multibleu=True,smoothing_function=cc.method3)
  M.train()
  with open(savestr+"hyps",'w') as f:
    hyps = [' '.join(x) for x in hyps]
    f.write('\n'.join(hyps))
  with open(savestr+"refs",'w') as f:
    refstr = []
    for r in refs:
      r = [' '.join(x) for x in r]
      refstr.append('\n'.join(r))
    f.write('\n'.join(refstr))
  return bleu
'''
def validate(M,DS):
  data = DS.val_batches
  cc = SmoothingFunction()
  M.eval()
  refs = []
  hyps = []
  for x in data:
    sources, targets = DS.pad_batch(x,targ=False)
    sources = Variable(sources.cuda(),volatile=True)
    M.zero_grad()
    logits = M(sources,None)
    logits = torch.max(logits.data.cpu(),2)[1]
    logits = [list(x) for x in logits]
    hyp = [x[:x.index(1)+1] if 1 in x else x for x in logits]
    hyp = [[DS.vocab[x] for x in y] for y in hyp]
    hyps.extend(hyp)
    refs.extend(targets)
  print(len(refs),len(hyps))
  bleu = corpus_bleu(refs,hyps,emulate_multibleu=True,smoothing_function=cc.method3)
  M.train()
  with open("debugging_refs",'w') as f:
    refstr = []
    for r in refs:
      r = [' '.join(x) for x in r]
      refstr.append('\n'.join(r))
    f.write('\n'.join(refstr))
  return bleu
'''

def params():
  parser = argparse.ArgumentParser(description='none')
  # learning
  parser.add_argument('-dataset', type=str, default="data/valid.txt.ner.9ref")
  parser.add_argument('-m1', type=str)
  parser.add_argument('-m2', type=str)
  parser.add_argument('-ds1', type=str)
  parser.add_argument('-ds2', type=str)
  parser.add_argument('-savestr', type=str, default="valid_results/")

  args = parser.parse_args()
  return args
 
if __name__=="__main__":
  args = params() 
  M1,_ = torch.load(args.m1)
  M2,_ = torch.load(args.m2)
  M1.enc.flatten_parameters()
  M1.dec.flatten_parameters()
  M2.enc.flatten_parameters()
  M2.dec.flatten_parameters()
  DS1 = torch.load(args.ds1)
  DS2 = torch.load(args.ds2)
  DS1.new_data(args.dataset)
  data = DS1.new_batches
  args.savestr += "_" + ''.join([x for x in args.m1+args.m2 if x!='/'])
  cc = SmoothingFunction()
  M1.eval()
  M2.eval()
  refs = []
  hyps = []
  inters = []
  for x in data:
    sources, targets = DS1.pad_batch(x,targ=False)
    sources = Variable(sources.cuda(),volatile=True)
    M1.zero_grad()
    M2.zero_grad()
    logits = M1(sources,None)
    logits = torch.max(logits.data.cpu(),2)[1]
    logits = [list(x) for x in logits]
    hyp = [x[:x.index(1)] if 1 in x else x for x in logits]
    hyp = [[DS1.vocab[x] for x in y] for y in hyp]
    inters.extend(hyp)
    sources2 = (hyp,targets)
    s2, _ = DS2.pad_batch(sources2,targ=False)
    s2 = Variable(s2.cuda(),volatile=True)
    logits2 = M2(s2,None)
    logits2 = torch.max(logits2.data.cpu(),2)[1]
    logits2 = [list(x) for x in logits2]
    hyp2 = [x[:x.index(1)] if 1 in x else x for x in logits2]
    hyp2 = [[DS2.vocab[x] for x in y] for y in hyp2]
    hyps.extend(hyp2)
    refs.extend(targets)
  bleu = corpus_bleu(refs,hyps,emulate_multibleu=True,smoothing_function=cc.method3)
  print(bleu)
  with open(args.savestr+"hyps",'w') as f:
    hyps = [' '.join(x) for x in hyps]
    f.write('\n'.join(hyps))
  with open(args.savestr+"refs",'w') as f:
    refs = ['\t'.join([' '.join(x) for x in y]) for y in refs]
    f.write('\n'.join(refs))
