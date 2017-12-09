import sys
import os
import argparse
import torch
from itertools import product
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from verb_preprocess import load_data
from arguments import s2s_inter as parseParams
from time import time

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
    self.vdec = nn.LSTM(args.hsz,args.hsz,num_layers=args.layers,batch_first=True)
    self.venc = nn.LSTM(args.hsz,args.hsz,num_layers=args.layers,bidirectional=True,batch_first=True)
    self.vgen = nn.Linear(args.hsz,args.verbvsz)
    self.vlinin = nn.Linear(args.hsz,args.hsz,bias=False)
    self.vlinout = nn.Linear(args.hsz*2,args.hsz,bias=False)

    # attention stuff
    self.linin = nn.Linear(args.hsz,args.hsz,bias=False)
    self.dvlinin = nn.Linear(args.hsz,args.hsz,bias=False)
    self.sm = nn.Softmax()
    self.linout = nn.Linear(args.hsz*2,args.hsz,bias=False)
    self.tanh = nn.Tanh()
    self.drop = nn.Dropout(args.drop)
    self.pretrain = False
    self.beamsize = args.beamsize

  def forward(self,inp,out=None,verbs=None):
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
        prev = Variable(torch.cuda.LongTensor(inp.size(0),1).fill_(3))
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
    if self.pretrain:
      return voutputs

    _, vidxs = torch.max(voutputs,2)
    vembedding = self.vemb(vidxs)
    vencoding, _ = self.vdec(vembedding,(hx,cx))
    h = hx
    c = cx


    '''
    vmask = torch.cuda.ByteTensor(verbs.size(0),verbs.size(1),args.hsz).zero_()
    for i in range(verbs.size(0)):
      for j in range(verbs.size(1)):
        if verbs[i,j].data[0]==0:
          vmask[i,j,:].fill_(1)
    vencoding = torch.cat(vencoding,1)
    vencoding.data.masked_fill_(vmask,0)
    '''


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

      '''
      #attend on enc 
      q = self.linin(decout.squeeze(1)).unsqueeze(2)
      w = torch.bmm(enc,q).squeeze(2)
      w = self.sm(w)
      cc = torch.bmm(w.unsqueeze(1),enc)
      '''
      
      #attend on vencoding
      vq = self.dvlinin(decout.squeeze(1)).unsqueeze(2)
      vw = torch.bmm(vencoding,vq).squeeze(2)
      vw = self.sm(vw)
      vcc = torch.bmm(vw.unsqueeze(1),vencoding)

      
      #op = torch.cat((cc,decout,vcc),2)
      op = torch.cat((decout,vcc),2)
      op = self.drop(self.tanh(self.linout(op)))
      outputs.append(self.gen(op))

    outputs = torch.cat(outputs,1)
    return outputs


  def beamsearch(self,inp):
    inp = inp.unsqueeze(0)
    encenc = self.encemb(inp)
    enc,(he,ce) = self.enc(encenc)

    #enc hidden has bidirection so switch those to the features dim
    hx = torch.cat([he[0:he.size(0):2], he[1:he.size(0):2]], 2) 
    cx = torch.cat([ce[0:ce.size(0):2], ce[1:ce.size(0):2]], 2) 
    h = [torch.cat([he[0:he.size(0):2], he[1:he.size(0):2]], 2) for i in range(self.beamsize)]
    c = [torch.cat([ce[0:ce.size(0):2], ce[1:ce.size(0):2]], 2) for i in range(self.beamsize)]
    encbeam = [enc for i in range(self.beamsize)]

    #decode verbs
    #op = Variable(torch.cuda.FloatTensor(inp.size(0),self.args.hsz).zero_())
    op = Variable(torch.FloatTensor(inp.size(0),self.args.hsz).zero_())
    voutputs = []
    vencoding = []

    outp = self.args.vmaxlen

    for i in range(outp): 
      if i == 0:
        #prev = Variable(torch.cuda.LongTensor(inp.size(0),1).fill_(3))
        prev = Variable(torch.LongTensor(inp.size(0),1).fill_(3))
      else:
        prev = self.vgen(op).max(2)
        prev = prev[1]
        if prev[0,0].data[0]==1:
          break
        #op = op.squeeze(1)
          

      #dembedding = self.vemb(prev)
      #decin = torch.cat((dembedding.squeeze(1),op),1).unsqueeze(1)
      decin = self.vemb(prev)
      decout, (hx,cx) = self.vdec(decin,(hx,cx))
      vencoding.append(decout)

      #attend on enc 
      q = self.vlinin(decout.squeeze(1)).unsqueeze(2)
      w = torch.bmm(enc,q).squeeze(2)
      w = self.sm(w)
      cc = torch.bmm(w.unsqueeze(1),enc)
      op = torch.cat((cc,decout),2)
      op = self.drop(self.tanh(self.vlinout(op)))
      voutputs.append(self.vgen(op))

    #vencoding_ = torch.cat(vencoding,1)
    #vencoding = [vencoding_ for i in range(self.beamsize)]
    voutputs = torch.cat(voutputs,1)
    _, vidxs = torch.max(voutputs,2)
    verbs = vidxs.data
    vembedding = self.vemb(vidxs)
    vencoding_, _ = self.vdec(vembedding)
    vencoding = [vencoding_ for _ in range(self.beamsize)]

    ops = [Variable(torch.FloatTensor(1,1,self.args.hsz).zero_()) for i in range(self.beamsize)]
    prev = [Variable(torch.LongTensor(1,1).fill_(3)) for i in range(self.beamsize)]
    #ops = [Variable(torch.cuda.FloatTensor(1,1,self.args.hsz).zero_()) for i in range(self.beamsize)]
    #prev = [Variable(torch.cuda.LongTensor(1,1).fill_(3)) for i in range(self.beamsize)]
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
        dembedding = self.decemb(prev[j])
        decin = torch.cat((dembedding.squeeze(1),ops[j].squeeze(1)),1).unsqueeze(1)
        decout, (hx,cx) = self.dec(decin,(h[j],c[j]))

        '''
        #attend on enc 
        q = self.linin(decout.squeeze(1)).unsqueeze(2)
        #q = decout.view(decout.size(0),decout.size(2),decout.size(1))
        w = torch.bmm(enc,q).squeeze(2)
        w = self.sm(w)
        cc = torch.bmm(w.unsqueeze(1),enc)
        '''
        
        #attend on vencoding
        vq = self.dvlinin(decout.squeeze(1)).unsqueeze(2)
        vw = torch.bmm(vencoding[j],vq).squeeze(2)
        vw = self.sm(vw)
        attns[j].append(vw)
        vcc = torch.bmm(vw.unsqueeze(1),vencoding[j])

        
        op = torch.cat((decout,vcc),2)
        #op = torch.cat((cc,decout,vcc),2)
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
    return done[topscore], doneattns[topscore], verbs

def validate(M,DS,args):
  data = DS.val_batches
  cc = SmoothingFunction()
  M.eval()
  M = M.cpu()
  refs = []
  hyps = []
  i=0
  print(time())
  for x in data:
    if args.debug:
      sources, targets, verbs = DS.pad_batch(x,targ=False)
    else:
      sources, targets = DS.pad_batch(x,targ=False,v=False)
    sources = Variable(sources.cpu(),volatile=True)
    M.zero_grad()
    logits = []
    attns = []
    inter = []
    for s in sources:
      l,a,v = M.beamsearch(s)
      logits.append(l)
      attns.append(a)
      inter.append(v)
    hyp = [x[:x.index(1)] if 1 in x else x for x in logits]
    hyp = [[DS.vocab[x] for x in y] for y in hyp]
    for b in inter:
      ihyp = [[DS.verb_vocab[x] for x in list(y)] for y in b]
      print(ihyp)
    print(hyp)
    print(targets)
    hyps.extend(hyp)
    refs.extend(targets)
  bleu = corpus_bleu(refs,hyps,emulate_multibleu=True,smoothing_function=cc.method3)
  M = M.cuda()
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

def train(M,DS,args,optimizer):
  data = DS.train_batches
  weights = torch.cuda.FloatTensor(args.vsz).fill_(1)
  weights[0] = 0
  criterion = nn.CrossEntropyLoss(weights)
  trainloss = []
  for x in data:
    sources, targets, verbs = DS.pad_batch(x)
    sources = Variable(sources.cuda())
    targets = Variable(targets.cuda())
    verbs = Variable(verbs,requires_grad=False)
    M.zero_grad()
    logits= M(sources,targets,verbs)
    logits = logits.view(-1,logits.size(2))
    targets = targets.view(-1)

    loss = criterion(logits, targets)
    loss.backward()
    trainloss.append(loss.data.cpu()[0])
    optimizer.step()

    if len(trainloss)%100==99: print(trainloss[-1])
  return sum(trainloss)/len(trainloss)

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


def main(args):
  DS = torch.load(args.datafile)
  if args.debug:
    DS.train_batches= DS.train_batches[:2]
    DS.val_batches= DS.train_batches[:2]
    args.epochs = 100
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
  M.endtok = DS.vocab.index("<eos>")
  M.punct = [DS.vocab.index(t) for t in ['.','!','?']]
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
    if args.debug:
      if epoch%2==1:
        b = validate(M,DS,args)
        print("valid bleu ",b)
    else: 
      b = validate(M,DS,args)
      print("valid bleu ",b)
      torch.save((M,optimizer),args.savestr+args.epoch+"_bleu-"+str(b))

if __name__=="__main__":
  args = parseParams()
  main(args)
