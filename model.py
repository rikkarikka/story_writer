import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes, num_layers,
                 num_dir, batch_size, emb_dim,
                 dropout, net_type, prevecs=None, embfix=False):
    super().__init__()
    self.num_layers = num_layers
    self.num_dir = num_dir
    self.batch_size = batch_size
    self.hidden_size = hidden_size
    self.emb_dim = emb_dim
    self.emb = nn.Embedding(input_size, emb_dim)

    self.egru = nn.GRU(emb_dim, hidden_size, num_layers=num_layers,
                        batch_first=True,bidirectional=(num_dir==2),
                        dropout=dropout)
    self.dgru = nn.GRU(emb_dim*2, hidden_size*2, num_layers=num_layers, 
                        batch_first=True,dropout=dropout)

    self.attn_in = nn.Linear(hidden_size*2,hidden_size*2,bias=False)
    self.attn_out = nn.Linear(hidden_size*4,emb_dim,bias=False)
    self.sm = nn.Softmax()
    self.drop = nn.Dropout(dropout)
    self.generate = nn.Linear(hidden_size*4,num_classes)
    self.tanh = nn.Tanh()

  def get_hx(self,size):
    hx = Variable(torch.FloatTensor(self.num_layers*self.num_dir,
                                            size, self.hidden_size).zero_())
    if int(torch.cuda.is_available()) == 1:
        hx.data = hx.data.cuda()
    return hx

  def forward(self, inp, outp):
    he = self.get_hx(inp.size(0))
    enc,h = self.egru(inp,he)
    #enc hidden has bidirection so switch those to the features dim
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) 
    outputs = []
    attentions = []
    op = Variable(torch.cuda.FloatTensor(inp.size(0),self.emb_dim).zero_(), requires_grad=False)
    for i in range(outp.size(1)): 
      o = outp[:,i,:]
      decin = torch.cat((o,op),1).unsqueeze(1)
      print(decin.size())
      dec,h = self.dgru(decin,h)
      #attn
      q = dec.squeeze(1)
      print(q.size())
      q = self.attn_in(q).unsqueeze(2)
      attn_w = torch.bmm(enc,q).squeeze(2)
      #TODO:masking goes here
      attn_w = self.sm(attn_w)
      attn = attn_w.unsqueeze(1)
      attn = torch.bmm(attn,enc)
      attn = attn.squeeze(1)
      attnq = torch.cat((attn,q),1).squeeze()
      print(attnq.size())
      print(self.hidden_size)
      op = self.tanh(self.attn_out(attnq))
      op = self.drop(op)
      #generate
      gen = self.drop(self.generate(attnq))
      outputs.append(gen)
      attentions.append(attn_w)
    outputs = torch.stack(outputs)
    return outputs, attentions
