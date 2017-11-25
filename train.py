import sys
import os
import argparse
import torch
from itertools import product
from torch import nn
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from data import load_data
from generate import genmax

def train(M,DS,args,optimizer):
  data = DS.train_batches
  weights = torch.cuda.FloatTensor(args.vsz).fill_(1)
  weights[0] = 0
  criterion = nn.CrossEntropyLoss(weights)
  trainloss = []
  for sources,targets in data:
    sources = Variable(sources.cuda())
    targets = Variable(targets.cuda())
    M.zero_grad()
    outs = M(sources,targets)
    logits = M.gen(outs)
    targets = targets[:,1:].contiguous()
    logits = logits[:,:targets.size(1),:].contiguous()
    logits = logits.view(-1,logits.size(2))
    targets = targets.view(-1)
    loss = criterion(logits, targets)
    loss.backward()
    trainloss.append(loss.data.cpu()[0])
    optimizer.step()
  return sum(trainloss)/len(trainloss)

def main():
  args = parseParams()
  if args.model=="s2s":
    from s2s import model
  elif args.model=="s2sattn":
    from s2sattn import model
  elif args.model=="eventattn":
    from eventattn import model
    args.hsz = 300
  else:
    print("Bad model option");exit()
  args.gentargs = True
  args.savestr = "saved_models/"+args.model+"/"
  try:
    os.stat(args.savestr)
  except:
    os.mkdir(args.savestr)
  if args.debug:
    args.eventvocab = "two.txt"
    args.bsz = 2
    DS = load_data(args.bsz,debug=True)
  else:
    try:  
      DS = torch.load(args.datafile)
    except:
      DS = load_data(args.bsz)
      torch.save(DS,args.datafile)
  args.vsz = len(DS.itos_targets)
  args.start = DS.stoi_targets["<start>"]
  if args.resume:
    M, optimizer = torch.load(args.resume)
    M.enc.flatten_parameters()
    M.dec.flatten_parameters()
    print("resuming from ",args.resume)
  else:
    M = model(args).cuda()
    print(M)
    if args.model == "eventattn":
      init_vocab(M,DS,args)
    optimizer = torch.optim.Adam(M.parameters(), lr=args.lr)
  for epoch in range(args.epochs):
    args.epoch = str(epoch)
    trainloss = train(M,DS,args,optimizer)
    print("train loss epoch",epoch,trainloss)
    vloss, vbleu = validate(M,DS,args)
    print("valid loss ",vloss)
    print("valid bleu ",vbleu)
    savestr = args.savestr+"epoch_"+args.epoch+"vbleu_"+str(vbleu)[:4]+"-vloss_"+str(vloss)[:4]+".pt"
    torch.save((M,optimizer),savestr)

def parseParams():
    parser = argparse.ArgumentParser(description='none')
    # learning
    parser.add_argument('-layers', type=int, default=2, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-drop', type=float, default=0.3, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-bsz', type=int, default=50, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-hsz', type=int, default=500, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-maxlen', type=int, default=75, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]') #
    parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 100]') #
    #parser.add_argument('-savedir', type=str, default="saved_models/s2s/")
    parser.add_argument('-datafile', type=str, default="data/datafile.pt")
    parser.add_argument('-model', type=str, default="s2sattn")
    parser.add_argument('-eventvocab', type=str, default="data/verbnet_categories.txt")
    parser.add_argument('-debug', action="store_true")
    parser.add_argument('-resume', type=str, default=None)
 
    args = parser.parse_args()
    return args

main()
