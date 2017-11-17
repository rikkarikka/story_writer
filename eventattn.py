import torch
from torch import nn
from torch.autograd import Variable

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

  def mkevents(self,em):
    self.em = Variable(em.unsqueeze(0).expand(self.args.bsz,em.size(0),em.size(1)))

  def forward(self,inp,out=None):
    enc,(h,c) = self.enc(inp)

    #calculate event attentions
    eW,_ = torch.max(torch.bmm(enc,self.em.permute(0,2,1)),1)
    eW = self.sm(eW)
    
    eWW = eW.unsqueeze(2).expand(eW.size(0),eW.size(1),self.args.hsz)
    Vattn = torch.mul(eWW,self.em)

    #enc hidden has bidirection so switch those to the features dim
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) 
    c = torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2) 

    #decode
    op = Variable(torch.cuda.FloatTensor(inp.size(0),self.args.hsz).zero_())
    if out is None:
      outp = self.args.maxlen
    else:
      outp = out.size(1)

    outputs = []
    attns = []
    for i in range(outp): 
      if i == 0:
        prev = Variable(torch.cuda.LongTensor(self.args.bsz,1).fill_(self.args.start))
      else:
        if out is None:
          prev = self.gen(op).max(2)
          prev = prev[1]
        else:
          prev = out[:,i]
        op = op.squeeze()
          
      dembedding = self.decemb(prev)
      decin = torch.cat((dembedding.squeeze(),op),1).unsqueeze(1)
      decout, (h,c) = self.dec(decin,(h,c))

      #attend on events 
      q = self.linin(decout.squeeze(1)).unsqueeze(2)
      w = torch.bmm(Vattn,q).squeeze(2)
      w = self.sm(w)
      attns.append(w.unsqueeze(2))
      cc = torch.bmm(w.unsqueeze(1),Vattn).squeeze(1)
      op = torch.cat((cc,decout.squeeze(1)),1)
      op = self.drop(self.tanh(self.linout(op)).unsqueeze(1))
      outputs.append(op)

    outputs = torch.cat(outputs,1)
    attns = torch.cat(attns,2)
    return outputs,attns,eW


