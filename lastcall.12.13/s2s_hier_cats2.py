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
    self.encemb = nn.Embedding(args.svsz,args.hsz,padding_idx=0)
    self.enc = nn.LSTM(args.hsz,args.hsz//2,bidirectional=True,num_layers=args.layers,batch_first=True)
    self.internoun = nn.Embedding(args.nvsz,args.hsz)
    self.interverb = nn.Embedding(args.vvsz,args.hsz)
    self.inter = nn.LSTM(args.hsz*3,args.hsz,num_layers=args.layers,batch_first=True)
    self.noungen = nn.Linear(args.hsz,args.nvsz)
    self.verbgen = nn.Linear(args.hsz,args.vvsz)
    self.decemb = nn.Embedding(args.vsz,args.hsz,padding_idx=0)
    self.dec = nn.LSTM(args.hsz*4,args.hsz,num_layers=args.layers,batch_first=True)
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
    ogenc = enc

    #enc hidden has bidirection so switch those to the features dim
    he = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) 
    ce = torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2) 
    h = [torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) for i in range(self.beamsize)]
    c = [torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2) for i in range(self.beamsize)]
    encbeam = [enc for i in range(self.beamsize)]
    ogencbeam = [enc for i in range(self.beamsize)]
    
    hinter = []
    cinter = []
    interout = []
    nprev = []
    vprev = []
    nstack = []
    vstack = []
    inters = []
    interstack = []
    for _ in range(self.beamsize):
      inters.append([])
    for i in range(self.beamsize):
      hinter.append(Variable(he.data).contiguous())
      cinter.append(Variable(ce.data).contiguous())
      iout, (hinter[i], cinter[i]) = self.inter(Variable(torch.cuda.FloatTensor(1,1,self.args.hsz*3).zero_()),
                                                (hinter[i],cinter[i]))
      interout.append(iout)

      nbest = self.noungen(iout)
      _,nmax = torch.max(nbest,2)
      vbest = self.verbgen(iout)
      _,vmax = torch.max(vbest,2)
      inters[i].append((nmax,vmax))
      nemb = self.internoun(nmax)
      vemb = self.interverb(vmax)
      #print(nemb.size(),vemb.size())
      nstack.append(nemb)
      vstack.append(vemb)
      #print(enc.size(),nemb.size(),vemb.size())
      interstack.append(torch.cat((nemb,vemb),2))
    nstack = torch.stack(nstack,0)
    vstack = torch.stack(vstack,0)
    

    if self.args.cuda:
      changeidx = torch.cuda.LongTensor(inp.size(0)).zero_()
      infostack = torch.cuda.LongTensor(inp.size(0)).fill_(1)
    else:
      changeidx = torch.LongTensor(inp.size(0),1).zero_()
      infostack = torch.LongTensor(inp.size(0)).fill_(1)


    ops = [Variable(torch.cuda.FloatTensor(1,1,self.args.hsz).zero_()) for i in range(self.beamsize)]
    prev = [Variable(torch.cuda.LongTensor(1,1).fill_(3)) for i in range(self.beamsize)]
    beam = [[] for x in range(self.beamsize)]
    scores = [0]*self.beamsize
    sents = [0]*self.beamsize
    done = []
    donescores = []
    doneencbeam = []
    doneattns = []
    doneinters = []
    donecounts = []
    attns = [[] for _ in range(self.beamsize)]
    changecounts = [0 for _ in range(self.beamsize)]
    for i in range(self.args.maxlen):
      if not beam:
        break
      tmp = []
      for j in range(len(beam)):
        changej = 0
        for t in self.punct:
          changej += prev[j].squeeze().data.eq(t)[0]
        if changej>0:
          changecounts[j] += 1
          iin = torch.cat((ops[j], interstack[j]),2)
          iout, (hinter[j], cinter[j]) = self.inter(iin,(hinter[j],cinter[j]))
          interout[j] = iout
          nbest = self.noungen(iout)
          #print(nbest.size())
          _,nmax = torch.max(nbest,2)
          vbest = self.verbgen(iout)
          _,vmax = torch.max(vbest,2)
          inters[j].append((nmax,vmax))
          nemb = self.internoun(nmax)
          vemb = self.interverb(vmax)
          interstack[j] = torch.cat((nemb,vemb),2)
          #print(ogenc.size(),nemb.size(),vemb.size())
        dembedding = self.decemb(prev[j].view(1,1))
        #print(dembedding.size(),ops[j].size(),interstack[j].size())
        decin = torch.cat((dembedding.squeeze(1),ops[j].squeeze(0),interstack[j].squeeze(0)),1).unsqueeze(1)
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
        donectr = 0
        k = -1
        while donectr<self.beamsize:
          k+=1
          pdat = pidx[k].data[0]
          if pdat == 2:
            continue
          else:
            tmp.append((vals[k].data[0]+scores[j],pidx[k],j,hx,cx,op,sents[j]))
            donectr+=1
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
      newencbeam = []
      added = 0
      newinters = []
      newcounts = []
      j = 0
      while added < len(beam):
        v,pidx,beamidx,hx,cx,op,sentlen = tmp[j]
        pdat = pidx.data[0]
        new = beam[beamidx]+[pdat]
        if pdat in self.punct:
          sentlen+=1
        if pdat == self.endtok or sentlen>4:
          if new not in done:
            done.append(new)
            donescores.append(v)
            doneattns.append(attns[beamidx])
            doneencbeam.append(encbeam[beamidx])
            doneinters.append(inters[beamidx])
            donecounts.append(changecounts[beamidx])
            added += 1
            j+=1
          else:
            j+=1
            continue
        else:
          if new not in newbeam:
            newsents.append(sentlen)
            newinters.append([x for x in inters[beamidx]])
            newbeam.append(new)
            newscore.append(v)
            newh.append(hx)
            newcounts.append(changecounts[beamidx])
            newc.append(cx)
            newhinter.append(hinter[beamidx])
            newcinter.append(cinter[beamidx])
            newops.append(op)
            newprev.append(pidx)
            newattns.append([x for x in attns[beamidx]])
            newinterout.append(interout[beamidx])
            newencbeam.append(encbeam[beamidx])
            added += 1
          j+=1
      beam = newbeam 
      changecounts = newcounts
      prev = newprev
      encbeam = newencbeam
      scores = newscore
      sents = newsents
      interout = newinterout
      h = newh
      c = newc
      ops = newops
      inters = newinters
    if len(done)==0:
      done.extend(beam)
      donescores.extend(scores)
      doneattns.extend(attns)
      doneinters = inters
      donecounts = changecounts
    donescores = [x/len(done[i]) for i,x in enumerate(donescores)]
    print(donescores)
    topscore =  donescores.index(max(donescores))
    print(donecounts[topscore])
    return done[topscore], doneattns[topscore], doneinters[topscore]

      
  def forward(self,inp,verbs,nouns,out=None):
    encenc = self.encemb(inp)
    enc,(h,c) = self.enc(encenc)

    #enc hidden has bidirection so switch those to the features dim
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) 
    c = torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2) 

    #embed nouns & verbs
    nemb = self.internoun(nouns)
    vemb = self.interverb(verbs)
    info = torch.cat((nemb,vemb),2)

    hinter = []
    cinter = []
    nlogits = []
    vlogits = []
    interout = []
    for i in range(inp.size(0)):
      hinter.append(Variable(h.data[:,i,:]).unsqueeze(1).contiguous())
      cinter.append(Variable(c.data[:,i,:]).unsqueeze(1).contiguous())
      iout, (hinter[i], cinter[i]) = self.inter(Variable(torch.cuda.FloatTensor(1,1,self.args.hsz*3).zero_()),
                                                (hinter[i],cinter[i]))
      interout.append(iout)
      nlogits.append([self.noungen(iout)])
      vlogits.append([self.verbgen(iout)])


    if self.args.cuda:
      changeidx = torch.cuda.LongTensor(inp.size(0)).zero_()
      infostack = torch.cuda.LongTensor(inp.size(0)).fill_(1)
    else:
      changeidx = torch.LongTensor(inp.size(0),1).zero_()
      infostack = torch.LongTensor(inp.size(0)).fill_(1)

    interstack = torch.stack(([info[i,infostack[i],:] for i in range(inp.size(0))]),0)
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
        changed = False
        for k in changes:
          k = k.cpu()[0]
          iin = torch.cat((op[k].unsqueeze(0), interstack[k].view(1,1,-1)),2)
          iout, (hinter[k], cinter[k]) = self.inter(iin,(hinter[k],cinter[k]))
          interout[k] = iout
          infostack[k] += 1
          nlogits[k].append(self.noungen(iout))
          vlogits[k].append(self.verbgen(iout))
          changed=True
        if changed:
          interstack = torch.stack(([info[i,infostack[i],:] for i in range(inp.size(0))]),0).squeeze(1)
        op = op.squeeze(1)
          

      dembedding = self.decemb(prev)
      decin = torch.cat((dembedding.squeeze(1),op,interstack),1).unsqueeze(1)
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
    nout = Variable(torch.cuda.FloatTensor(nouns.size(0),nouns.size(1),self.args.nvsz))
    vout = Variable(torch.cuda.FloatTensor(verbs.size(0),verbs.size(1),self.args.vvsz))
    for i in range(inp.size(0)):
      l = len(nlogits[i])
      nout[i,:l,:] = torch.cat(nlogits[i],1)
      l = len(vlogits[i])
      vout[i,:l,:] = torch.cat(vlogits[i],1)
    return outputs, vout, nout


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
    loss.backward()
    self.optimizer.step()
    return loss.data.cpu()[0]

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
    logits,vout,nout = M(sources,verbs,nouns,targets)

    loss = jl.calc(targets,nouns,verbs,logits,nout,vout)
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

if __name__=="__main__":
  args = parseParams()
  if "bland" in args.savestr:
    print("set save dir to hierarchical")
    exit()
  main(args)
