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
from arguments import s2s_bland as parseParams
from s2s_bland import model

'''
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

    # attention stuff
    self.linin = nn.Linear(args.hsz,args.hsz,bias=False)
    self.sm = nn.Softmax()
    self.linout = nn.Linear(args.hsz*2,args.hsz,bias=False)
    self.tanh = nn.Tanh()
    self.drop = nn.Dropout(args.drop)
    self.beamsize = args.beamsize

  def beamsearch(self,inp):
    #inp = inp.unsqueeze(0)
    encenc = self.encemb(inp)
    enc,(h,c) = self.enc(encenc)

    #enc hidden has bidirection so switch those to the features dim
    h = [torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) for i in range(self.beamsize)]
    c = [torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2) for i in range(self.beamsize)]
    encbeam = [enc for i in range(self.beamsize)]

    ops = [Variable(torch.cuda.FloatTensor(1,1,self.args.hsz).zero_()) for i in range(self.beamsize)]
    prev = [Variable(torch.cuda.LongTensor(1,1).fill_(3)) for i in range(self.beamsize)]
    beam = [[] for x in range(self.beamsize)]
    scores = [0]*self.beamsize
    sents = [0]*self.beamsize
    done = []
    donescores = []
    for i in range(self.args.maxlen):
      tmp = []
      for j in range(len(beam)):
        dembedding = self.decemb(prev[j].view(1,1))
        decin = torch.cat((dembedding.squeeze(1),ops[j].squeeze(1)),1).unsqueeze(1)
        decout, (hx,cx) = self.dec(decin,(h[j],c[j]))

        #attend on enc 
        q = self.linin(decout.squeeze(1)).unsqueeze(2)
        #q = decout.view(decout.size(0),decout.size(2),decout.size(1))
        w = torch.bmm(enc,q).squeeze(2)
        w = self.sm(w)
        cc = torch.bmm(w.unsqueeze(1),enc)
        op = torch.cat((cc,decout),2)
        op = self.drop(self.tanh(self.linout(op)))
      
        op2 = self.gen(op)
        op2 = op2.squeeze()
        probs = F.log_softmax(op2)
        vals, pidx = probs.topk(self.beamsize*2,0)
        vals = vals.squeeze()
        pidx = pidx.squeeze()
        for k in range(self.beamsize):
          tmp.append((vals[k].data[0]+scores[j],pidx[k],j,hx,cx,op))
      tmp.sort(key=lambda x: x[0],reverse=True)
      newbeam = []
      newscore = []
      newsents = []
      newh = []
      newc = []
      newops = []
      newprev = []
      added = 0
      j = 0
      while added < len(beam):
        v,pidx,beamidx,hx,cx,op = tmp[j]
        pdat = pidx.data[0]
        new = beam[beamidx]+[pdat]
        if pdat in self.punct:
          newsents.append(sents[beamidx]+1)
        else:
          newsents.append(sents[beamidx])
        if pdat == self.endtok or newsents[-1]>4:
          if new not in done:
            done.append(new)
            donescores.append(v)
            added += 1
        else:
          if new not in newbeam:
            newbeam.append(new)
            newscore.append(v)
            newh.append(hx)
            newc.append(cx)
            newops.append(op)
            newprev.append(pidx)
            added += 1
        j+=1
      beam = newbeam 
      prev = newprev
      scores = newscore
      sents = newsents
      h = newh
      c = newc
      ops = newops
    if len(done)==0:
      done.extend(beam)
      donescores.extend(scores)
    donescores = [x/len(done[i]) for i,x in enumerate(donescores)]
    topscore =  donescores.index(max(donescores))
    return done[topscore]
      
  def forward(self,inp,out=None):
    encenc = self.encemb(inp)
    enc,(h,c) = self.enc(encenc)

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
        op = op.squeeze(1)
          

      dembedding = self.decemb(prev)
      decin = torch.cat((dembedding.squeeze(1),op),1).unsqueeze(1)
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
'''

def validate(M,DS,args):
  print(args.valid)
  data = DS.val_data(args.valid)
  cc = SmoothingFunction()
  M.eval()
  refs = []
  hyps = []
  weights = torch.cuda.FloatTensor(args.vsz).fill_(1)
  weights[0] = 0
  criterion = nn.CrossEntropyLoss(weights)
  trainloss = []
  for sources,targets in data:
    sources = Variable(sources.cuda(),volatile=True)
    targets = Variable(targets.cuda(),volatile=True)
    M.zero_grad()
    logits = M(sources,targets,val=True)
      
    loss = criterion(logits.view(-1,logits.size(2)),targets.view(-1))
    trainloss.append(loss.data.cpu()[0])
  return sum(trainloss)/len(trainloss)

def main(args):
  DS = torch.load(args.datafile)
  if args.debug:
    args.bsz=2
    DS.train = DS.train[:2]

  args.vsz = DS.vsz
  args.svsz = DS.svsz
  output = open("losses_blandval.txt",'w')
  output.write(args.datafile+'\n')
  for mname in [x for x in os.listdir(args.savestr) if x[0].isdigit()]:
    M,optimizer = torch.load(args.savestr+"/"+mname)
    M.enc.flatten_parameters()
    M.dec.flatten_parameters()
    M.endtok = DS.vocab.index("<eos>")
    M.punct = [DS.vocab.index(t) for t in ['.','!','?'] if t in DS.vocab]
    b = validate(M,DS,args)

    print(mname + "valid loss ",b)
    output.write(mname + '\t' + str(b) + '\n')
  output.close()

if __name__=="__main__":
  args = parseParams()
  main(args)
