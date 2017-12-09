import sys
import os
import argparse
import torch
from itertools import product
from torch import nn
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from verb_preprocess import load_data
from arguments import s2s_inter as parseParams

class model(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.args = args
    
    # encoder decoder stuff
    self.encemb = nn.Embedding(args.svsz,args.hsz,padding_idx=0)
    self.enc = nn.LSTM(args.hsz,args.hsz//2,bidirectional=True,num_layers=args.layers,batch_first=True)
    self.decemb = nn.Embedding(args.vsz,args.hsz,padding_idx=0)
    self.dec = nn.LSTM(args.hsz*2,args.hsz,num_layers=args.layers,batch_first=True)
    self.gen = nn.Linear(args.hsz,args.vsz)

    self.vemb = nn.Embedding(args.verbvsz,args.hsz,padding_idx=0)
    self.vdec = nn.LSTM(args.hsz*2,args.hsz,num_layers=args.layers,batch_first=True)
    self.vgen = nn.Linear(args.hsz,args.verbvsz)
    self.vlinin = nn.Linear(args.hsz,args.hsz,bias=False)
    self.vlinout = nn.Linear(args.hsz*2,args.hsz,bias=False)

    # attention stuff
    self.linin = nn.Linear(args.hsz,args.hsz,bias=False)
    self.dvlinin = nn.Linear(args.hsz,args.hsz,bias=False)
    self.sm = nn.Softmax()
    self.linout = nn.Linear(args.hsz*3,args.hsz,bias=False)
    self.tanh = nn.Tanh()
    self.drop = nn.Dropout(args.drop)
    self.pretrain = False

  def forward(self,inp,out=None,verbs=None):
    encenc = self.encemb(inp)
    enc,(h,c) = self.enc(encenc)

    #enc hidden has bidirection so switch those to the features dim
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) 
    c = torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2) 

    #decode verbs
    op = Variable(torch.cuda.FloatTensor(inp.size(0),self.args.hsz).zero_())
    voutputs = []
    vencoding = []

    if verbs is None:
      outp = self.args.vmaxlen
    else:
      outp = verbs.size(1)

    for i in range(outp): 
      if i == 0:
        prev = Variable(torch.cuda.LongTensor(inp.size(0),1).fill_(3))
      else:
        if verbs is None:
          prev = self.vgen(op).max(2)
          prev = prev[1]
        else:
          prev = verbs[:,i-1].unsqueeze(1)
        op = op.squeeze(1)
          

      dembedding = self.vemb(prev)
      decin = torch.cat((dembedding.squeeze(1),op),1).unsqueeze(1)
      decout, (h,c) = self.vdec(decin,(h,c))
      vencoding.append(decout)

      #attend on enc 
      q = self.vlinin(decout.squeeze(1)).unsqueeze(2)
      w = torch.bmm(enc,q).squeeze(2)
      w = self.sm(w)
      cc = torch.bmm(w.unsqueeze(1),enc)
      op = torch.cat((cc,decout),2)
      op = self.drop(self.tanh(self.vlinout(op)))
      voutputs.append(self.vgen(op))

    if self.pretrain:
      voutputs = torch.cat(voutputs,1)
      return voutputs

    vencoding= torch.cat(vencoding,1)


    #decode text
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
        op = op.squeeze(1)
          

      dembedding = self.decemb(prev)
      decin = torch.cat((dembedding.squeeze(1),op),1).unsqueeze(1)
      decout, (h,c) = self.dec(decin,(h,c))

      #attend on enc 
      q = self.linin(decout.squeeze(1)).unsqueeze(2)
      w = torch.bmm(enc,q).squeeze(2)
      w = self.sm(w)
      cc = torch.bmm(w.unsqueeze(1),enc)
      
      #attend on vencoding
      vq = self.dvlinin(decout.squeeze(1)).unsqueeze(2)
      vw = torch.bmm(vencoding,vq).squeeze(2)
      vw = self.sm(vw)
      vcc = torch.bmm(vw.unsqueeze(1),vencoding)

      
      op = torch.cat((cc,decout,vcc),2)
      op = self.drop(self.tanh(self.linout(op)))
      outputs.append(self.gen(op))

    outputs = torch.cat(outputs,1)
    voutputs = torch.cat(voutputs,1)
    return outputs,voutputs


def validate(M,DS,args):
  data = DS.val_batches
  cc = SmoothingFunction()
  M.eval()
  refs = []
  hyps = []
  for x in data:
    sources, targets = DS.pad_batch(x,targ=False,v=False)
    sources = Variable(sources.cuda(),volatile=True)
    M.zero_grad()
    logits,_ = M(sources,None,None)
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

def pretrain(M,DS,args,optimizer):
  M.pretrain=True
  data = DS.train_batches
  weights2 = torch.cuda.FloatTensor(args.verbvsz).fill_(1)
  weights2[0] = 0
  criterion2 = nn.CrossEntropyLoss(weights2)
  trainloss = []
  for x in data:
    sources, targets,verbs = DS.pad_batch(x)
    sources = Variable(sources)
    targets = Variable(targets)
    verbs = Variable(verbs)
    M.zero_grad()
    vlogits = M(sources,out=None,verbs=verbs)
    vlogits = vlogits.view(-1,vlogits.size(2))
    verbs = verbs.view(-1)
    loss = criterion2(vlogits,verbs)
    loss.backward()
    trainloss.append(loss.data.cpu()[0])
    optimizer.step()

    if len(trainloss)%100==99:
      print(trainloss[-1])
  M.pretrain=False
  return sum(trainloss)/len(trainloss)

def train(M,DS,args,optimizer):
  data = DS.train_batches
  weights = torch.cuda.FloatTensor(args.vsz).fill_(1)
  weights[0] = 0
  criterion = nn.CrossEntropyLoss(weights)
  weights2 = torch.cuda.FloatTensor(args.verbvsz).fill_(1)
  weights2[0] = 0
  criterion2 = nn.CrossEntropyLoss(weights2)
  trainloss = []
  verbloss = []
  for x in data:
    sources, targets,verbs = DS.pad_batch(x)
    sources = Variable(sources)
    targets = Variable(targets)
    verbs = Variable(verbs)
    M.zero_grad()
    logits,vlogits = M(sources,targets,verbs)
    logits = logits.view(-1,logits.size(2))

    targets = targets.view(-1)
    vlogits = vlogits.view(-1,vlogits.size(2))
    verbs = verbs.view(-1)
    loss = criterion(logits, targets) + criterion2(vlogits,verbs)
    loss.backward()
    trainloss.append(loss.data.cpu()[0])
    optimizer.step()

    if len(trainloss)%100==99:
      print(trainloss[-1])
  return sum(trainloss)/len(trainloss)

def main(args):
  DS = torch.load(args.datafile)
  if args.debug:
    args.bsz=2
    DS.train = DS.train[:2]
    DS.valid= DS.valid[:2]

  args.vsz = DS.vsz
  args.svsz = DS.svsz
  args.verbvsz = DS.vvsz
  if args.resume:
    M,optimizer = torch.load(args.resume)
    M.enc.flatten_parameters()
    M.dec.flatten_parameters()
    e = args.resume.split("/")[-1] if "/" in args.resume else args.resume
    e = e.split('_')[0]
    e = int(e)+1
  else:
    M = model(args).cuda()
    optimizer = torch.optim.Adam(M.parameters(), lr=args.lr)
    e=0
  print(M)
  print(args.datafile)
  print(args.savestr)
  for epoch in range(args.pretrainepochs):
    trainloss = pretrain(M,DS,args,optimizer)
    print("train loss epoch",epoch,trainloss)
  for epoch in range(e,args.epochs):
    args.epoch = str(epoch)
    trainloss = train(M,DS,args,optimizer)
    print("train loss epoch",epoch,trainloss)
    b = validate(M,DS,args)
    print("valid bleu ",b)
    torch.save((M,optimizer),args.savestr+args.epoch+"_bleu-"+str(b))

if __name__=="__main__":
  args = parseParams()
  main(args)
