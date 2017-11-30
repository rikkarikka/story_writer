import sys
import os
import argparse
import torch
from itertools import product
from torch import nn
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from preprocess import load_data
from arguments import sxs as parseParams

class model(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.args = args
    
    # encoder decoder stuff
    self.encemb = nn.Embedding(args.svsz,args.hsz,padding_idx=0)
    self.enc = nn.LSTM(args.hsz,args.hsz//2,bidirectional=True,num_layers=args.layers,batch_first=True)
    self.inter = nn.LSTM(args.hsz,args.hsz,num_layers=args.layers,batch_first=True)
    self.decemb = nn.Embedding(args.vsz,args.hsz,padding_idx=0)
    self.dec = nn.LSTM(args.hsz*3,args.hsz,num_layers=args.layers,batch_first=True)
    self.gen = nn.Linear(args.hsz,args.vsz)

    # attention stuff
    self.linin = nn.Linear(args.hsz,args.hsz,bias=False)
    self.sm = nn.Softmax()
    self.linout = nn.Linear(args.hsz*2,args.hsz,bias=False)
    self.tanh = nn.Tanh()
    self.drop = nn.Dropout(args.drop)
    self.endtok = -1

  def forward(self,inp,out=None):
    encenc = self.encemb(inp)
    enc,(h,c) = self.enc(encenc)

    #enc hidden has bidirection so switch those to the features dim
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) 
    c = torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2) 

    hinter = []
    cinter = []
    interout = []
    for i in range(inp.size(0)):
      hinter.append(Variable(h.data[:,i,:]).unsqueeze(1).contiguous())
      cinter.append(Variable(c.data[:,i,:]).unsqueeze(1).contiguous())
      iout, (hinter[i], cinter[i]) = self.inter(Variable(torch.cuda.FloatTensor(1,1,self.args.hsz).zero_()),
                                                (hinter[i],cinter[i]))
      interout.append(iout)


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
        for i in range(inp.size(0)):
          if prev[i].data.eq(self.endtok).sum()==1:
            iout, (hinter[i], cinter[i]) = self.inter(interout[i],(hinter[i],cinter[i]))
            interout[i] = iout

          
        op = op.squeeze(1)
          
      dembedding = self.decemb(prev)
      interstack = torch.cat(interout,0)
      decin = torch.cat((dembedding.squeeze(1),op,interstack.squeeze(1)),1).unsqueeze(1)
      decout, (h,c) = self.dec(decin,(h,c))

      #attend on enc 
      q = self.linin(decout.squeeze(1)).unsqueeze(2)
      #q = decout.view(decout.size(0),decout.size(2),decout.size(1))
      w = torch.bmm(enc,q).squeeze(2)
      w = self.sm(w)
      cc = torch.bmm(w.unsqueeze(1),enc)
      op = torch.cat((cc,decout),2)
      op = self.drop(self.tanh(self.linout(op)))
      outputs.append(self.gen(op))

    outputs = torch.cat(outputs,1)
    return outputs


def validate(M,DS,args):
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
  with open(args.savestr+"hyps"+args.epoch,'w') as f:
    hyps = [' '.join(x) for x in hyps]
    f.write('\n'.join(hyps))
  with open(args.savestr+"refs",'w') as f:
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
    sources, targets = DS.pad_batch(x)
    sources = Variable(sources.cuda())
    targets = Variable(targets.cuda())
    M.zero_grad()
    logits = M(sources,targets)
    logits = logits.view(-1,logits.size(2))
    targets = targets.view(-1)

    loss = criterion(logits, targets)
    loss.backward()
    trainloss.append(loss.data.cpu()[0])
    optimizer.step()

    if len(trainloss)%100==99: print(trainloss[-1])
  return sum(trainloss)/len(trainloss)

def main():
  args = parseParams()
  DS = torch.load(args.datafile)
  if args.debug:
    args.bsz=2
    DS.train = DS.train[:2]
    DS.valid= DS.valid[:2]

  args.vsz = DS.vsz
  args.svsz = DS.svsz
  if args.resume:
    M,optimizer = torch.load(args.resume)
    M.enc.flatten_parameters()
    M.dec.flatten_parameters()
    M.inter.flatten_parameters()
    e = args.resume.split("/")[-1] if "/" in args.resume else args.resume
    e = e.split('_')[0]
    e = int(e)+1
  else:
    optimizer = torch.optim.Adam(M.parameters(), lr=args.lr)
    M = model(args).cuda()
  M.endtok = DS.stoi["."]
  print(M)
  print('end tok idx: ',M.endtok)
  for epoch in range(e,args.epochs):
    args.epoch = str(epoch)
    trainloss = train(M,DS,args,optimizer)
    print("train loss epoch",epoch,trainloss)
    b = validate(M,DS,args)
    print("valid bleu ",b)
    torch.save((M,optimizer),args.savestr+args.epoch+"_bleu-"+str(b))

if __name__=="__main__":
  main()
