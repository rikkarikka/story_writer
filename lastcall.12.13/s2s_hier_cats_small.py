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

class model(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.args = args
    
    # encoder decoder stuff
    self.encemb = nn.Embedding(args.svsz,args.esz,padding_idx=0)
    self.enc = nn.LSTM(args.esz,args.hsz//2,bidirectional=True,num_layers=args.layers,batch_first=True)
    self.internoun = nn.Embedding(args.nvsz,args.esz)
    self.interverb = nn.Embedding(args.vvsz,args.esz)
    self.inter = nn.LSTM(args.hsz+(args.esz*2),args.hsz,num_layers=args.layers,batch_first=True)
    self.noungen = nn.Linear(args.hsz,args.nvsz)
    self.verbgen = nn.Linear(args.hsz,args.vvsz)
    self.decemb = nn.Embedding(args.vsz,args.esz,padding_idx=0)
    self.dec = nn.LSTM(args.hsz+args.esz,args.hsz,num_layers=args.layers,batch_first=True)
    self.gen = nn.Linear(args.hsz,args.vsz)

    # attention stuff
    self.linin = nn.Linear(args.hsz,args.esz*2,bias=False)
    self.ilinin = nn.Linear(args.hsz,args.hsz,bias=False)
    self.sm = nn.Softmax()
    self.linout = nn.Linear(args.hsz+(args.esz*2),args.hsz,bias=False)
    self.ilinout = nn.Linear(args.hsz*2,args.hsz,bias=False)
    self.tanh = nn.Tanh()
    self.drop = nn.Dropout(args.drop)

    #params
    self.punct = -1
    self.beamsize = args.beamsize

  def forward(self,inp,verbs,nouns,out):
    encenc = self.encemb(inp)
    enc,(h,c) = self.enc(encenc)

    #enc hidden has bidirection so switch those to the features dim
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) 
    c = torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2) 

    #embed nouns & verbs
    nemb = self.internoun(nouns)
    vemb = self.interverb(verbs)
    info = torch.cat((nemb,vemb),2)
    #print('info',info.size())

    hinter = Variable(h.data)
    cinter = Variable(c.data)
    catattn = Variable(torch.cuda.FloatTensor(inp.size(0),1,self.args.esz*2).zero_())
    
    nout = []
    vout = []
    s = 0
    def change_inter(intercatattn,hinter,cinter,s):
      previnter = info[:,s,:].unsqueeze(1)
      #print(previnter.size(),intercatattn.size())
      iin = torch.cat((previnter,intercatattn),2)
      iout, (hinter, cinter) = self.inter(iin,(hinter,cinter))

      #attend on enc 
      q = self.ilinin(iout.squeeze(1)).unsqueeze(2)
      w = torch.bmm(enc,q).squeeze(2)
      w = self.sm(w)
      cc = torch.bmm(w.unsqueeze(1),enc)
      op = torch.cat((cc,iout),2)
      catattn = self.drop(self.tanh(self.ilinout(op)))
      n = self.noungen(catattn)
      v = self.verbgen(catattn)
      nextinter = torch.cat((n,v),2)

      return nextinter, catattn, hinter, cinter, n, v

    for i in range(info.size(1)):
      nextinter, catattn, hinter, cinter, n, v = change_inter(catattn, hinter, cinter,i)
      nout.append(n)
      vout.append(v)
      
    outputs = []
    for i in range(out.size(1)):
      if i == 0:
        prev = Variable(torch.cuda.LongTensor(inp.size(0),1).fill_(3))
        dcatattn = Variable(torch.cuda.FloatTensor(inp.size(0),1,self.args.esz*2).zero_())
      else:
        prev = out[:,i-1].unsqueeze(1)

      pemb = self.decemb(prev)
      #print(pemb.size(),dcatattn.size())
      decin = torch.cat((pemb,dcatattn),2)
      decout, (h, c) = self.dec(decin,(h,c))

      #attend on verbs
      q = self.linin(decout.squeeze(1)).unsqueeze(2)
      w = torch.bmm(info,q).squeeze(2)
      w = self.sm(w)
      cc = torch.bmm(w.unsqueeze(1),info)
      op = torch.cat((cc,decout),2)
      dcatattn = self.drop(self.tanh(self.linout(op)))
      outputs.append(self.gen(dcatattn))

    outputs = torch.cat(outputs,1)
    nout = torch.cat(nout,1)
    vout = torch.cat(vout,1)
    #print(outputs.size(),nout.size(),vout.size())
    return outputs, nout, vout

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
    e = args.resume.split("/")[-1] if "/" in args.resume else args.resume
    e = e.split('_')[0]
    e = int(e)+1
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
  for epoch in range(e,args.epochs):
    args.epoch = str(epoch)
    trainloss = train(M,DS,args,jl)
    print("train loss epoch",epoch,trainloss)
    torch.save((M,jl),args.savestr+args.epoch)
    #vloss = valid(M,DS,args,jl)

if __name__=="__main__":
  args = parseParams()
  if "bland" in args.savestr:
    print("set save dir to hierarchical")
    exit()
  main(args)
