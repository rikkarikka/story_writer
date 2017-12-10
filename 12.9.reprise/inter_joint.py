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
import copy

class surface(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.args = args

    self.encemb = nn.Embedding(args.svsz,args.hsz,padding_idx=0)
    self.enc = nn.LSTM(args.hsz,args.hsz//2,bidirectional=True,num_layers=args.layers,batch_first=True)

    self.vemb = nn.Embedding(args.verbvsz,args.hsz,padding_idx=0)
    self.vdec = nn.LSTM(args.hsz,args.hsz,num_layers=args.layers,batch_first=True)
    self.vgen = nn.Linear(args.hsz,args.verbvsz)

    self.decemb = nn.Embedding(args.vsz,args.hsz,padding_idx=0)
    self.dec = nn.LSTM(args.hsz*2,args.hsz,num_layers=args.layers,batch_first=True)
    self.gen = nn.Linear(args.hsz,args.vsz)
  
    self.linin = nn.Linear(args.hsz,args.hsz,bias=False)
    self.linout = nn.Linear(args.hsz*3,args.hsz,bias=False)

    self.vlinin = nn.Linear(args.hsz,args.hsz,bias=False)
    self.vlinout = nn.Linear(args.hsz*2,args.hsz,bias=False)
    self.dvlinin = nn.Linear(args.hsz,args.hsz,bias=False)

    self.sm = nn.Softmax()
    self.tanh = nn.Tanh()
    self.drop = nn.Dropout(args.drop)

    self.beamsize = args.beamsize
    self.pretrain = False

  def forward(self,verbs,inp,mask,out=None):
    encenc = self.encemb(inp)
    enc,(h,c) = self.enc(encenc)

    #enc hidden has bidirection so switch those to the features dim
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) 
    c = torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2) 

    voutputs = []

    decin = self.vemb(verbs)
    decout, _ = self.vdec(decin,(h,c))
    qs = []
    for i in range(decout.size(1)):
      qs.append(self.vlinin(decout[:,i,:]))
    qs = torch.stack(qs,2)
    ws = torch.bmm(enc,qs)
    w = []
    for i in range(ws.size(2)):
      w.append(self.sm(ws[:,:,i]))
    w = torch.stack(w,1)
    w = torch.bmm(w,enc)
    op = torch.cat((w,decout),2)
    for i in range(w.size(1)):
      t = op[:,i,:]
      t = self.vlinout(t)
      t = self.drop(self.tanh(t))
      #voutputs.append(self.vgen(t))
      voutputs.append(t)
    voutputs = torch.stack(voutputs,1)
   
    vout = []
    for i in range(voutputs.size(1)):
      vout.append(self.vgen(voutputs[:,i,:]))
    vout = torch.stack(vout,1)
    if self.pretrain:
      return vout

    #decode text
    if self.args.cuda:
      op = Variable(torch.cuda.FloatTensor(inp.size(0),self.args.hsz).zero_())
    else:
      op = Variable(torch.FloatTensor(inp.size(0),self.args.hsz).zero_())
    outputs = []
    attns = []

    if out is None:
      outp = self.args.maxlen
    else:
      outp = out.size(1)

    for i in range(outp): 
      if i == 0:
        if self.args.cuda:
          prev = Variable(torch.cuda.LongTensor(inp.size(0),1).fill_(3))
        else:
          prev = Variable(torch.LongTensor(inp.size(0),1).fill_(3))
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
      vw = torch.bmm(voutputs,vq).squeeze(2)
      vw.data[mask].fill_(-float('inf'))
      vw = self.sm(vw)
      attns.append(vw)
      vcc = torch.bmm(vw.unsqueeze(1),voutputs)

      op = torch.cat((cc,decout,vcc),2)
      op = self.drop(self.tanh(self.linout(op)))
      outputs.append(self.gen(op))

    outputs = torch.cat(outputs,1)
    return outputs, attns, vout

  def beamsearch(self,inp):
    encenc = self.encemb(inp)
    enc,(h,c) = self.enc(encenc)

    #enc hidden has bidirection so switch those to the features dim
    he = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) 
    ce = torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2) 
    h = [he for _ in range(self.beamsize)]
    c = [ce for _ in range(self.beamsize)]
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
    for i in range(self.args.vmaxlen):
      tmp = []
      for j in range(len(beam)):
        decin = self.vemb(prev[j].view(1,1))
        decout, (hx,cx) = self.vdec(decin,(h[j],c[j]))

        #attend on enc
        vq = self.vlinin(decout.squeeze(1)).unsqueeze(2)
        vw = torch.bmm(enc,vq).squeeze(2)
        vw = self.sm(vw)
        vcc = torch.bmm(vw.unsqueeze(1),enc)

        op = torch.cat((decout,vcc),2)
        op = self.drop(self.tanh(self.vlinout(op)))
      
        op2 = self.vgen(op)
        op2 = op2.squeeze()
        probs = F.log_softmax(op2)
        vals, pidx = probs.topk(self.beamsize*2,0)
        vals = vals.squeeze()
        pidx = pidx.squeeze()
        for k in range(self.beamsize):
          tmp.append((vals[k].data[0]+scores[j],pidx[k],j,hx,cx))
      tmp.sort(key=lambda x: x[0],reverse=True)
      newbeam = []
      newscore = []
      newsents = []
      newh = []
      newc = []
      newprev = []
      added = 0
      j = 0
      while added < self.beamsize:
        v,pidx,beamidx,hx,cx = tmp[j]
        pdat = pidx.data[0]
        new = beam[beamidx]+[pdat]
        if pdat in self.punct:
          newsents.append(sents[beamidx]+1)
        else:
          newsents.append(sents[beamidx])
        #if pdat == self.vendtok or newsents[-1]>4:
        if pdat == self.vendtok and i>self.args.vminlen:
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
            newprev.append(pidx)
            added += 1
        j+=1
      beam = newbeam 
      prev = newprev
      scores = newscore
      sents = newsents
      h = newh
      c = newc
    if len(done)==0:
      done.extend(beam)
      donescores.extend(scores)
    donescores = [x/len(done[i]) for i,x in enumerate(donescores)]
    topscore = donescores.index(max(donescores))
    verbs = done[topscore]
    verbs = [1]+verbs
    if self.args.cuda:
      verbs = Variable(torch.cuda.LongTensor(verbs).unsqueeze(0))
    else:
      verbs = Variable(torch.LongTensor(verbs).unsqueeze(0))
    venc = self.vemb(verbs)
    voutputs,_ = self.vdec(venc,(he,ce))
    h = [he for _ in range(self.beamsize)]
    c = [ce for _ in range(self.beamsize)]

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
        vw = torch.bmm(voutputs,vq).squeeze(2)
        vw = self.sm(vw)
        attns[j].append(vw)
        vcc = torch.bmm(vw.unsqueeze(1),voutputs)

        #attend on enc
        vq = self.vlinin(decout.squeeze(1)).unsqueeze(2)
        vw = torch.bmm(enc,vq).squeeze(2)
        vw = self.sm(vw)
        cc = torch.bmm(vw.unsqueeze(1),enc)

        op = torch.cat((cc,decout,vcc),2)
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
      attns = newattns
    if len(done)==0:
      done.extend(beam)
      donescores.extend(scores)
      doneattns = attns
    donescores = [x/len(done[i]) for i,x in enumerate(donescores)]
    topscore =  donescores.index(max(donescores))
    return done[topscore], doneattns[topscore]

def train(S,DS,args,optimizer):
  weights = torch.cuda.FloatTensor(args.vsz).fill_(1)
  weights[0] = 0
  criterion = nn.CrossEntropyLoss(weights)
  weights2 = torch.cuda.FloatTensor(DS.vvsz).fill_(1)
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
    mask = verbs.data.eq(0)
    targets = Variable(targets.cuda(),requires_grad=False)
    S.zero_grad()
    if S.pretrain:
      vout = S(verbs,sources,mask,targets)
    else:
      logits, attns, vout = S(verbs,sources,mask,targets)
    vout = vout[:,:vout.size(1)-1,:].contiguous()
    verbs = verbs[:,1:].contiguous()
    vout = vout.view(-1,vout.size(2))
    verbs = verbs.view(-1)
    if S.pretrain:
      loss = criterion2(vout,verbs)
    else: 
      logits = logits.view(-1,logits.size(2))
      targets = targets.view(-1)
      loss = criterion(logits, targets) + criterion2(vout,verbs)
    loss.backward()
    trainloss.append(loss.data.cpu()[0])
    optimizer.step()

    if len(trainloss)%100==99: print(trainloss[-1])
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
    S,optimizer = torch.load(args.resume)
    S.enc.flatten_parameters()
    S.dec.flatten_parameters()
    e = args.resume.split("/")[-1] if "/" in args.resume else args.resume
    e = e.split('_')[0]
    e = int(e)+1
  else:
    S = surface(args).cuda()
    Sopt = torch.optim.Adam(S.parameters(), lr=args.lr)
    e=0
    pe=0
  S.endtok = DS.vocab.index("<eos>")
  S.punct = [DS.vocab.index(t) for t in ['.','!','?']]
  print(S)
  print(args.datafile)
  print(args.savestr)
  S.pretrain = True
  for epoch in range(pe,args.pretrainepochs):
    trainloss = train(S,DS,args,Sopt)
    print("train loss epoch",epoch,trainloss)
    torch.save((S,Sopt),args.savestr+"_pretrain_"+str(epoch))
  print("done pretraining")
  S.pretrain = False
  S.args.vminlen = 5
  for epoch in range(e,args.epochs):
    args.epoch = str(epoch)
    trainloss = train(S,DS,args,Sopt)
    print("train loss epoch",epoch,trainloss)
    torch.save((S,Sopt),args.savestr+args.epoch)

if __name__=="__main__":
  args = parseParams()
  main(args)
