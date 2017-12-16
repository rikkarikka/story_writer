import sys
import os
import argparse
import torch
from itertools import product
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from preprocess_new import load_data
from s2s_bland import model
from arguments import s2s_bland as parseParams
import pickle
  
def draw(inters,surface,attns,args):
  for i in range(len(inters)):
    try:
      os.mkdir(args.savestr+"attns/")
    except:
      pass
    with open(args.savestr+"attns/"+args.epoch+"-"+str(i),'wb') as f:
      pickle.dump((inters[i],surface[i],attns[i].data.cpu().numpy()),f)

def validate(S,DS,args,m):
  print(m,args.valid)
  S.beamsize = args.beamsize
  data = DS.new_data(args.valid)
  cc = SmoothingFunction()
  S.eval()
  refs = []
  hyps = []
  attns = []
  inters = []
  titles = []
  for sources,targets in data:
    title = [DS.itos[x] for x in sources[0]]
    titles.append(" ".join(title))
    sources = Variable(sources,requires_grad=False)
    logits = []
    attn = []
    l = S.beamsearch(sources)
    logits.append(l)
    hyp = [DS.vocab[x] for x in logits[0]]
    hyps.append(hyp)
    print(' '.join(hyp))
    refs.append(targets)
    assert(len(hyps)==len(refs))
  draw(inters,hyps,attns,args)
  bleu = corpus_bleu(refs,hyps,emulate_multibleu=True,smoothing_function=cc.method3)
  print(bleu)
  S.train()
  with open(args.savestr+"titles",'w') as f:
    f.write("\n".join(titles))
  with open(args.savestr+"hyps"+m+"-bleu_"+str(bleu),'w') as f:
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

def main(args,m):
  DS = torch.load(args.datafile)
  DS.args = args
  print(m)
  args.epoch = m
  S,_ = torch.load(args.savestr+m)
  if not args.cuda:
    print('move to cpu')
    S = S.cpu()
  S.dec.flatten_parameters()
  S.enc.flatten_parameters()
  S.args = args
  S.endtok = DS.vocab.index("<eos>")
  validate(S,DS,args,m)

if __name__=="__main__":
  args = parseParams()
  if args.vmodel:
    models = [args.vmodel]
  else:
    models = [x for x in os.listdir(args.savestr) if x[0].isdigit()]
    models.sort(reverse=True)
  for m in models:
    main(args,m)
