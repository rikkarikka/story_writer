import torch
from torch import nn
from torch.autograd import Variable

class s2s(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.args = args
    self.enc = nn.LSTM(300,args.hsz//2,bidirectional=True,num_layers=args.layers,batch_first=True)
    self.decemb = nn.Embedding(args.vsz,args.hsz,padding_idx=0)
    self.dec = nn.LSTM(args.hsz*2,args.hsz,num_layers=args.layers,batch_first=True)
    self.gen = nn.Linear(args.hsz,args.vsz)

  def forward(self,inp,out=None):
    enc,(h,c) = self.enc(inp)

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
      outputs.append(decout)
      op = decout
    outputs = torch.cat(outputs,1)
    return outputs


