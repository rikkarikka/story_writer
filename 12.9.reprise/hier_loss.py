import sys
import os
import argparse
import torch
from itertools import product
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from inter_preprocess_new import load_data
from s2s_hierarchical import model
from arguments import s2s_hierarchical as parseParams
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
  data = DS.new_data(args.valid,targ=True)
  cc = SmoothingFunction()
  S.eval()
  refs = []
  hyps = []
  attns = []
  inters = []
  titles = []
  weights = torch.cuda.FloatTensor(args.vsz).fill_(1)
  weights[0] = 0
  criterion = nn.CrossEntropyLoss(weights)
  trainloss = []
  for sources,targets in data:
    sources = Variable(sources.cuda(),volatile=True)
    targets = Variable(targets.cuda(),volatile=True)
    S.zero_grad()
    logits = S(sources)
    logits = logits.repeat(targets.size(0),1,1)
    size = min(logits.size(1),targets.size(1))
    targets = targets[:,:size].contiguous()
    logits = logits[:,:size,:].contiguous()
    loss = criterion(logits.view(-1,logits.size(2)),targets.view(-1))
    trainloss.append(loss.data.cpu()[0])
  return sum(trainloss)/len(trainloss)

def main(args,m):
  DS = torch.load(args.datafile)
  DS.args = args
  print(m)
  args.epoch = m
  args.vsz = DS.vsz
  args.svsz = DS.svsz
  S,_ = torch.load(args.savestr+m)
  if not args.cuda:
    print('move to cpu')
    S = S.cpu()
  S.dec.flatten_parameters()
  S.enc.flatten_parameters()
  S.inter.flatten_parameters()
  S.args = args
  S.endtok = DS.vocab.index("<eos>")
  S.vendtok = DS.verb_vocab.index("<eos>")
  S.punct = [DS.vocab.index(t) for t in ['.','!','?']]
  l = validate(S,DS,args,m)
  print(l)
  return l


if __name__=="__main__":
  args = parseParams()
  if args.vmodel:
    m = args.vmodel
    main(args,m)
  else:
    output = open("losses_blandval.txt",'w')
    output.write(args.datafile+'\n')
    models = [x for x in os.listdir(args.savestr) if x[0].isdigit()]
    models.sort(reverse=True)
    for m in models:
      l = main(args,m)
      output.write(m + '\t' + str(l) + '\n')
