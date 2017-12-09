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
from arguments import s2s_inter as parseParams
import pickle

class inter(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.args = args
    
    # encoder decoder stuff
    self.encemb = nn.Embedding(args.svsz,args.hsz,padding_idx=0)
    self.enc = nn.LSTM(args.hsz,args.hsz//2,bidirectional=True,num_layers=args.layers,batch_first=True)

    self.vemb = nn.Embedding(args.verbvsz,args.hsz,padding_idx=0)
    self.vdec = nn.LSTM(args.hsz,args.hsz,num_layers=args.layers,batch_first=True)
    self.vgen = nn.Linear(args.hsz,args.verbvsz)
 
    self.vlinin = nn.Linear(args.hsz,args.hsz,bias=False)
    self.vlinout = nn.Linear(args.hsz*2,args.hsz,bias=False)
    self.sm = nn.Softmax()
    self.tanh = nn.Tanh()
    self.drop = nn.Dropout(args.drop)

  def encode(self,inp,hc):
    inp = Variable(inp,requires_grad=False)
    emb = self.vemb(inp)
    out, (h,c) = self.vdec(emb,hc)
    return out, (h,c)

  def forward(self,inp,verbs=None):
    encenc = self.encemb(inp)
    enc,(he,ce) = self.enc(encenc)

    #enc hidden has bidirection so switch those to the features dim
    h = torch.cat([he[0:he.size(0):2], he[1:he.size(0):2]], 2) 
    c = torch.cat([ce[0:ce.size(0):2], ce[1:ce.size(0):2]], 2) 
    hx = torch.cat([he[0:he.size(0):2], he[1:he.size(0):2]], 2) 
    cx = torch.cat([ce[0:ce.size(0):2], ce[1:ce.size(0):2]], 2) 

    #decode verbs
    #op = Variable(torch.cuda.FloatTensor(inp.size(0),self.args.hsz).zero_())
    voutputs = []
    vencoding = []

    if verbs is None:
      outp = self.args.vmaxlen
    else:
      outp = verbs.size(1)

    for i in range(outp): 
      if i == 0:
        if self.args.cuda:
          prev = Variable(torch.cuda.LongTensor(inp.size(0),1).fill_(3))
        else:
          prev = Variable(torch.LongTensor(inp.size(0),1).fill_(3))
      else:
        if verbs is None:
          prev = self.vgen(op).max(2)
          prev = prev[1]
        else:
          prev = verbs[:,i-1].unsqueeze(1)
        #op = op.squeeze(1)
          

      decin = self.vemb(prev)
      #dembedding = self.vemb(prev)
      #decin = torch.cat((dembedding.squeeze(1),op),1).unsqueeze(1)
      decout, (h,c) = self.vdec(decin,(h,c))
      #vencoding.append(decout)

      #attend on enc 
      q = self.vlinin(decout.squeeze(1)).unsqueeze(2)
      w = torch.bmm(enc,q).squeeze(2)
      w = self.sm(w)
      cc = torch.bmm(w.unsqueeze(1),enc)
      op = torch.cat((cc,decout),2)
      op = self.drop(self.tanh(self.vlinout(op)))
      voutputs.append(self.vgen(op))

    voutputs = torch.cat(voutputs,1)
    return voutputs, (hx,cx)

class surface(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.args = args
    self.decemb = nn.Embedding(args.vsz,args.hsz,padding_idx=0)
    self.dec = nn.LSTM(args.hsz*2,args.hsz,num_layers=args.layers,batch_first=True)
    self.gen = nn.Linear(args.hsz,args.vsz)
  
    self.dvlinin = nn.Linear(args.hsz,args.hsz,bias=False)
    self.linout = nn.Linear(args.hsz*2,args.hsz,bias=False)
    self.sm = nn.Softmax()
    self.tanh = nn.Tanh()
    self.drop = nn.Dropout(args.drop)

    self.beamsize = args.beamsize

  def forward(self,vencoding,hc,mask,out=None):
    h,c = hc
    #decode text
    if self.args.cuda:
      op = Variable(torch.cuda.FloatTensor(vencoding.size(0),self.args.hsz).zero_())
    else:
      op = Variable(torch.FloatTensor(vencoding.size(0),self.args.hsz).zero_())
    outputs = []
    attns = []

    if out is None:
      outp = self.args.maxlen
    else:
      outp = out.size(1)

    for i in range(outp): 
      if i == 0:
        if self.args.cuda:
          prev = Variable(torch.cuda.LongTensor(vencoding.size(0),1).fill_(3))
        else:
          prev = Variable(torch.LongTensor(vencoding.size(0),1).fill_(3))
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
      
      #attend on vencoding
      vq = self.dvlinin(decout.squeeze(1)).unsqueeze(2)
      vw = torch.bmm(vencoding,vq).squeeze(2)
      vw.data[mask].fill_(-float('inf'))
      vw = self.sm(vw)
      attns.append(vw)
      vcc = torch.bmm(vw.unsqueeze(1),vencoding)

      op = torch.cat((decout,vcc),2)
      op = self.drop(self.tanh(self.linout(op)))
      outputs.append(self.gen(op))

    outputs = torch.cat(outputs,1)
    return outputs, attns

  def beamsearch(self,vencoding,hc):
    hx,cx = hc

    #enc hidden has bidirection so switch those to the features dim
    #h = [torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) for i in range(self.beamsize)]
    #c = [torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2) for i in range(self.beamsize)]
    h = [hx for _ in range(self.beamsize)]
    c = [cx for _ in range(self.beamsize)]


    if self.args.cuda:
      ops = [Variable(torch.cuda.FloatTensor(1,1,self.args.hsz).zero_()) for i in range(self.beamsize)]
      prev = [Variable(torch.cuda.LongTensor(1,1).fill_(3)) for i in range(self.beamsize)]
    else:
      ops = [Variable(torch.FloatTensor(1,1,self.args.hsz).zero_()) for i in range(self.beamsize)]
      prev = [Variable(torch.LongTensor(1,1).fill_(3)) for i in range(self.beamsize)]
    beam = [[] for x in range(self.beamsize)]
    scores = [0]*self.beamsize
    sents = [0]*self.beamsize
    done = []
    donescores = []
    doneattns = []
    attns = [[] for x in range(self.beamsize)]
    for i in range(self.args.maxlen):
      tmp = []
      for j in range(len(beam)):
        dembedding = self.decemb(prev[j].view(1,1))
        decin = torch.cat((dembedding.squeeze(1),ops[j].squeeze(1)),1).unsqueeze(1)
        decout, (hx,cx) = self.dec(decin,(h[j],c[j]))

        #attend on vencoding
        vq = self.dvlinin(decout.squeeze(1)).unsqueeze(2)
        vw = torch.bmm(vencoding,vq).squeeze(2)
        #vw.data[mask].fill_(-float('inf'))
        vw = self.sm(vw)
        attns[j].append(vw)
        vcc = torch.bmm(vw.unsqueeze(1),vencoding)

        op = torch.cat((decout,vcc),2)
        #op = self.drop(self.tanh(self.linout(op)))
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
      newattns = []
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
            doneattns.append(attns[beamidx])
            added += 1
        else:
          if new not in newbeam:
            newbeam.append(new)
            newscore.append(v)
            newh.append(hx)
            newc.append(cx)
            newops.append(op)
            newprev.append(pidx)
            newattns.append(attns[beamidx])
            added += 1
        j+=1
      beam = newbeam 
      prev = newprev
      scores = newscore
      sents = newsents
      h = newh
      c = newc
      ops = newops
      newattns = attns
    if len(done)==0:
      done.extend(beam)
      donescores.extend(scores)
      doneattns = attns
    donescores = [x/len(done[i]) for i,x in enumerate(donescores)]
    topscore =  donescores.index(max(donescores))
    return done[topscore], doneattns[topscore]

def train(I,S,DS,args,optimizer):
  weights = torch.cuda.FloatTensor(args.vsz).fill_(1)
  weights[0] = 0
  criterion = nn.CrossEntropyLoss(weights)
  trainloss = []
  while True:
    x = DS.get_batch()
    if not x:
      break
    sources,targets,verbs = x
    sources = Variable(sources)
    predinter,(he,ce) = I.forward(sources)
    _, preds = torch.max(predinter,2)
    mask = torch.cuda.ByteTensor(preds.size()).zero_()
    for i in range(preds.size(0)):
      p = list(preds[i].data)
      if 1 in p[:-1]:
        mask[i,p.index(1)+1:].fill_(1)
    predenc,(hv,cv) = I.encode(preds.data,(he,ce))
    #h = torch.cat([he,hv],2)
    #c = torch.cat([ce,cv],2)
    h = hv; c = cv
    ihyp = [[DS.verb_vocab[x] for x in list(y)] for y in preds.data]
    targets = Variable(targets.cuda())
    S.zero_grad()
    logits, _= S(predenc,(h,c),mask,targets)
    logits = logits.view(-1,logits.size(2))
    targets = targets.view(-1)

    loss = criterion(logits, targets)
    loss.backward()
    trainloss.append(loss.data.cpu()[0])
    optimizer.step()

    if len(trainloss)%100==99: print(trainloss[-1])
  return sum(trainloss)/len(trainloss)

def pretrain(I,DS,args,optimizer):
  weights2 = torch.cuda.FloatTensor(args.verbvsz).fill_(1)
  weights2[0] = 0
  criterion2 = nn.CrossEntropyLoss(weights2)
  trainloss = []
  while True:
    x = DS.get_batch()
    if not x:
      break
    sources,targets,verbs = x
    sources = Variable(sources)
    verbs = Variable(verbs)
    I.zero_grad()
    vlogits,_ = I(sources,verbs=verbs)
    vlogits = vlogits.view(-1,vlogits.size(2))
    verbs = verbs.view(-1)
    loss = criterion2(vlogits,verbs)
    loss.backward()
    trainloss.append(loss.data.cpu()[0])
    optimizer.step()

    if len(trainloss)%100==99:
      print(trainloss[-1])
  return sum(trainloss)/len(trainloss)


def main(args):
  DS = torch.load(args.datafile)
  if args.debug:
    DS.train_batches= DS.train_batches[:2]
    DS.val_batches= DS.train_batches[:2]
    args.epochs = 1000
  args.vsz = DS.vsz
  args.svsz = DS.svsz
  args.verbvsz = DS.vvsz
  if args.resume:
    #TODO
    M,optimizer = torch.load(args.resume)
    M.enc.flatten_parameters()
    M.dec.flatten_parameters()
    e = args.resume.split("/")[-1] if "/" in args.resume else args.resume
    e = e.split('_')[0]
    e = int(e)+1
  elif args.resume_pre:
    I,Iopt = torch.load(args.resume_pre)
    I.enc.flatten_parameters()
    I.vdec.flatten_parameters()
    pe = args.resume_pre.split("/")[-1] if "/" in args.resume_pre else args.resume_pre
    pe = pe.split('_')[-1]
    pe = int(pe)+1
    S = surface(args).cuda()
    e=0
    Sopt = torch.optim.Adam(S.parameters(), lr=args.lr)

  else:
    I = inter(args).cuda()
    S = surface(args).cuda()
    Iopt = torch.optim.Adam(I.parameters(), lr=args.lr)
    Sopt = torch.optim.Adam(S.parameters(), lr=args.lr)
    e=0
    pe=0
  S.endtok = DS.vocab.index("<eos>")
  S.punct = [DS.vocab.index(t) for t in ['.','!','?']]
  print(I)
  print(S)
  print(args.datafile)
  print(args.savestr)
  for epoch in range(pe,args.pretrainepochs):
    trainloss = pretrain(I,DS,args,Iopt)
    print("train loss epoch",epoch,trainloss)
    torch.save((I,Iopt),args.savestr+"_pretrain_"+str(epoch))
  print("done pretraining")
  for epoch in range(e,args.epochs):
    args.epoch = str(epoch)
    trainloss = train(I,S,DS,args,Sopt)
    print("train loss epoch",epoch,trainloss)
    torch.save((I,S,Iopt,Sopt),args.savestr+args.epoch)

if __name__=="__main__":
  args = parseParams()
  main(args)
