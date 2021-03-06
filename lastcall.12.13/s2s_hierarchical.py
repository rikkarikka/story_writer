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
from arguments import s2s_hierarchical as parseParams

class model(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.args = args
    
    # encoder decoder stuff
    self.encemb = nn.Embedding(args.svsz,args.hsz,padding_idx=0)
    self.enc = nn.LSTM(args.hsz,args.hsz//2,bidirectional=True,num_layers=args.layers,batch_first=True)
    self.inter = nn.LSTM(args.hsz*2,args.hsz,num_layers=args.layers,batch_first=True)
    self.decemb = nn.Embedding(args.vsz,args.hsz,padding_idx=0)
    self.dec = nn.LSTM(args.hsz*3,args.hsz,num_layers=args.layers,batch_first=True)
    self.gen = nn.Linear(args.hsz,args.vsz)

    # attention stuff
    self.linin = nn.Linear(args.hsz,args.hsz,bias=False)
    self.sm = nn.Softmax()
    self.linout = nn.Linear(args.hsz*2,args.hsz,bias=False)
    self.tanh = nn.Tanh()
    self.drop = nn.Dropout(args.drop)

    #params
    self.punct = -1
    self.beamsize = args.beamsize

  def beamsearch(self,inp):
    #inp = inp.unsqueeze(0)
    encenc = self.encemb(inp)
    enc,(h,c) = self.enc(encenc)

    #enc hidden has bidirection so switch those to the features dim
    he = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) 
    ce = torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2) 
    h = [torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) for i in range(self.beamsize)]
    c = [torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2) for i in range(self.beamsize)]
    encbeam = [enc for i in range(self.beamsize)]
    
    hinter = []
    cinter = []
    interout = []
    for i in range(self.beamsize):
      hinter.append(Variable(he.data).contiguous())
      cinter.append(Variable(ce.data).contiguous())
      iout, (hinter[i], cinter[i]) = self.inter(Variable(torch.cuda.FloatTensor(1,1,self.args.hsz*2).zero_()),
                                                (hinter[i],cinter[i]))
      interout.append(iout)

    ops = [Variable(torch.cuda.FloatTensor(1,1,self.args.hsz).zero_()) for i in range(self.beamsize)]
    prev = [Variable(torch.cuda.LongTensor(1,1).fill_(3)) for i in range(self.beamsize)]
    beam = [[] for x in range(self.beamsize)]
    scores = [0]*self.beamsize
    sents = [0]*self.beamsize
    done = []
    donescores = []
    doneattns = []
    attns = [[] for _ in range(self.beamsize)]
    for i in range(self.args.maxlen):
      tmp = []
      for j in range(len(beam)):
        changej = 0
        for t in self.punct:
          changej += prev[j].squeeze().data.eq(t)[0]
        if changej>0:
          iin = torch.cat((ops[j], interout[j]),2)
          iout, (hinter[j], cinter[j]) = self.inter(iin,(hinter[j],cinter[j]))
          interout[j] = iout
        dembedding = self.decemb(prev[j].view(1,1))
        interstack = torch.cat(interout,0)
        decin = torch.cat((dembedding.squeeze(1),ops[j].squeeze(0),interstack[j]),1).unsqueeze(1)
        decout, (hx,cx) = self.dec(decin,(h[j],c[j]))

        #attend on enc 
        q = self.linin(decout.squeeze(1)).unsqueeze(2)
        #q = decout.view(decout.size(0),decout.size(2),decout.size(1))
        w = torch.bmm(enc,q).squeeze(2)
        w = self.sm(w)
        attns[j].append(w)
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
      newinterout = []
      newhinter = []
      newcinter = []
      newattns = []
      added = 0
      j = 0
      while added < len(beam):
        v,pidx,beamidx,hx,cx,op = tmp[j]
        pdat = pidx.data[0]
        new = beam[beamidx]+[pdat]
        sentlen = sents[beamidx]
        if pdat == 2:
          j+=1
          continue
        if pdat in self.punct:
          sentlen+=1
        if pdat == self.endtok or sentlen>4:
          if new not in done:
            done.append(new)
            donescores.append(v)
            doneattns.append(attns[beamidx])
            added += 1
          else:
            j+=1
            continue
        else:
          if new not in newbeam:
            newsents.append(sentlen)
            newbeam.append(new)
            newscore.append(v)
            newh.append(hx)
            newc.append(cx)
            newhinter.append(hinter[beamidx])
            newcinter.append(cinter[beamidx])
            newops.append(op)
            newprev.append(pidx)
            newattns.append(attns[beamidx])
            newinterout.append(interout[beamidx])
            added += 1
        j+=1
      beam = newbeam 
      prev = newprev
      scores = newscore
      sents = newsents
      interout = newinterout
      h = newh
      c = newc
      ops = newops
    if len(done)==0:
      done.extend(beam)
      donescores.extend(scores)
      doneattns.extend(attns)
    donescores = [x/len(done[i]) for i,x in enumerate(donescores)]
    print(donescores)
    topscore =  donescores.index(max(donescores))
    return done[topscore], doneattns[topscore]
      
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
      iout, (hinter[i], cinter[i]) = self.inter(Variable(torch.cuda.FloatTensor(1,1,self.args.hsz*2).zero_()),
                                                (hinter[i],cinter[i]))
      interout.append(iout)


    if self.args.cuda:
      changeidx = torch.cuda.LongTensor(inp.size(0)).zero_()
    else:
      changeidx = torch.LongTensor(inp.size(0),1).zero_()

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
        changeidx = changeidx.zero_()
        for t in self.punct:
          changeidx.add_(prev.squeeze().data.eq(t).long())
        changes = changeidx.nonzero()
        for k in changes:
          k = k.cpu()[0]
          iin = torch.cat((op[k].unsqueeze(0), interout[k]),2)
          iout, (hinter[k], cinter[k]) = self.inter(iin,(hinter[k],cinter[k]))
          interout[k] = iout
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

def train(M,DS,args,optimizer):
  weights = torch.cuda.FloatTensor(args.vsz).fill_(1)
  weights[0] = 0
  criterion = nn.CrossEntropyLoss(weights)
  trainloss = []
  while True:
    x = DS.get_batch()
    if not x:
      break
    # dont need verbs here
    sources,targets,_ = x
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

def main(args):
  DS = torch.load(args.datafile)
  if args.debug:
    args.bsz=2
    DS.train = DS.train[:2]

  args.vsz = DS.vsz
  args.svsz = DS.svsz
  if args.resume:
    M,optimizer = torch.load(args.resume)
    M.enc.flatten_parameters()
    M.inter.flatten_parameters()
    M.dec.flatten_parameters()
    e = args.resume.split("/")[-1] if "/" in args.resume else args.resume
    e = e.split('_')[0]
    e = int(e)+1
  else:
    M = model(args).cuda()
    optimizer = torch.optim.Adam(M.parameters(), lr=args.lr)
    e=0
  M.endtok = DS.vocab.index("<eos>")
  M.punct = [DS.vocab.index(t) for t in ['.','!','?'] if t in DS.vocab]
  print(M)
  print(args.datafile)
  print(args.savestr)
  for epoch in range(e,args.epochs):
    args.epoch = str(epoch)
    trainloss = train(M,DS,args,optimizer)
    print("train loss epoch",epoch,trainloss)
    torch.save((M,optimizer),args.savestr+args.epoch)

if __name__=="__main__":
  args = parseParams()
  main(args)
