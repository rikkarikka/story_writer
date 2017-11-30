import sys
import os
import argparse
import torch
from itertools import product
from torch import nn
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from preprocess import load_data, vecs

class model(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.args = args
    self.enc = nn.LSTM(300,args.hsz//2,bidirectional=True,num_layers=args.layers,batch_first=True)
    self.decemb = nn.Embedding(args.vsz,args.hsz,padding_idx=0)
    self.dec = nn.LSTM(args.hsz*2,args.hsz,num_layers=args.layers,batch_first=True)
    self.gen = nn.Linear(args.hsz,args.vsz)
    self.linin = nn.Linear(args.hsz,args.hsz)
    self.sm = nn.Softmax()
    self.linout = nn.Linear(args.hsz*2,args.hsz)
    self.tanh = nn.Tanh()
    self.drop = nn.Dropout(args.drop)

  def forward(self,inp,mask,out=None):
    enc,(h,c) = self.enc(inp)

    #enc hidden has bidirection so switch those to the features dim
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) 
    c = torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2) 

    #decode
    op = Variable(torch.cuda.FloatTensor(inp.size(0),self.args.hsz).zero_())
    outputs = []
    if out is None:
      outp = self.args.maxlen
    else:
      outp = out.size(1)

    for i in range(outp): 
      if i == 0:
        prev = Variable(torch.cuda.LongTensor(inp.size(0),1).fill_(3))
      else:
        if out is None:
          prev = self.gen(op).max(2)
          prev = prev[1]
        else:
          prev = out[:,i-1].unsqueeze(1)
        op = op.squeeze()
          
      dembedding = self.decemb(prev)
      decin = torch.cat((dembedding.squeeze(),op),1).unsqueeze(1)
      decout, (h,c) = self.dec(decin,(h,c))

      op = decout
      outputs.append(self.gen(decout))

      '''
      #attend on enc 
      #q = self.linin(decout.squeeze(1)).unsqueeze(2)
      q = decout.view(decout.size(0),decout.size(2),decout.size(1))
      w = torch.bmm(enc,q).squeeze(2)
      w.data.masked_fill_(mask, -float('inf'))
      w = self.sm(w)
      cc = torch.bmm(w.unsqueeze(1),enc)
      op = torch.cat((cc,decout),2)
      op = self.drop(self.tanh(self.linout(op)))
      outputs.append(self.gen(op))
      '''

    outputs = torch.cat(outputs,1)
    return outputs


def validate(M,DS,args):
  data = DS.val_batches
  weights = torch.cuda.FloatTensor(args.vsz).fill_(1)
  cc = SmoothingFunction()
  M.eval()
  refs = []
  hyps = []
  for x in data:
    sources, targets,mask = DS.vec_batch(x,targ=False)
    sources = Variable(sources.cuda(),volatile=True)
    M.zero_grad()
    logits = M(sources,mask,None)
    #outs = M(sources,None)
    #logits = M.gen(outs)
    logits = torch.max(logits.data.cpu(),2)[1]
    logits = [list(x) for x in logits]
    hyp = [x[:x.index(1)+1] if 1 in x else x for x in logits]
    hyp = [[DS.vocab[x] for x in y] for y in hyp]
    hyps.extend(hyp)
    refs.extend(targets)
  bleu = corpus_bleu(refs,hyps,emulate_multibleu=True,smoothing_function=cc.method3)
  M.train()
  with open("saved_models/s2s/hyps"+args.epoch,'w') as f:
    hyps = [' '.join(x) for x in hyps]
    f.write('\n'.join(hyps))
  try:
    os.stat("saved_models/s2s/refs")
  except:
    with open("saved_models/s2s/refs"+args.epoch,'w') as f:
      refstr = []
      for r in refs:
        r = [' '.join(x) for x in r]
        refstr.append('\n'.join(r))
      f.write('\n'.join(refstr))
  return bleu


def train(M,DS,args,optimizer):
  data = DS.train_batches
  weights = torch.cuda.FloatTensor(args.vsz).fill_(1)
  weights[0] = 0
  criterion = nn.CrossEntropyLoss(weights)
  trainloss = []
  for x in data:
    sources, targets,mask = DS.vec_batch(x)
    sources = Variable(sources.cuda())
    targets = Variable(targets.cuda())
    M.zero_grad()
    logits = M(sources,mask,targets)
    #outs = M(sources,targets)
    #logits = M.gen(outs)
    #targets = targets[:,1:].contiguous()
    #logits = logits[:,:targets.size(1),:].contiguous()
    logits = logits.view(-1,logits.size(2))
    targets = targets.view(-1)

    loss = criterion(logits, targets)
    loss.backward()
    trainloss.append(loss.data.cpu()[0])
    optimizer.step()
    '''
    _,maxes = torch.max(logits,1)
    maxes[targets.eq(0)]=1
    #print(maxes.eq(targets).sum())
    targets[targets.ne(0)]=1
    #print(targets.sum())
    '''
    if len(trainloss)%100==99: print(trainloss[-1])
  return sum(trainloss)/len(trainloss)

def main():
  args = parseParams()
  args.savestr = "saved_models/s2snoatt/"
  try:
    os.stat(args.savestr)
  except:
    os.mkdir(args.savestr)
  if args.debug:
    args.bsz=2
    args.datafile = "data/multiref_debug.pt"
  DS = torch.load(args.datafile)
  DS.mkbatches(args.bsz)
  args.vsz = DS.vsz
  M = model(args).cuda()
  print(M)
  optimizer = torch.optim.Adam(M.parameters(), lr=args.lr)
  for epoch in range(args.epochs):
    args.epoch = str(epoch)
    trainloss = train(M,DS,args,optimizer)
    print("train loss epoch",epoch,trainloss)
    b = validate(M,DS,args)
    print("valid bleu ",b)
    if not args.debug:
      torch.save((M,optimizer),args.savestr+args.epoch+"_bleu-"+str(b))

def parseParams():
    parser = argparse.ArgumentParser(description='none')
    # learning
    parser.add_argument('-layers', type=int, default=2, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-drop', type=float, default=0.3, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-bsz', type=int, default=100, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-hsz', type=int, default=500, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-maxlen', type=int, default=75, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]') #
    parser.add_argument('-epochs', type=int, default=30, help='number of epochs for train [default: 100]') #
    parser.add_argument('-datafile', type=str, default="data/multiref.pt")
    parser.add_argument('-debug', action="store_true")
    parser.add_argument('-resume', type=str, default=None)
 
    args = parser.parse_args()
    return args

main()
