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
    self.emd_dim = emb_dim
    self.emb = nn.Embedding(input_size, emb_dim)

    self.egru = nn.GRU(emb_dim, hidden_size, num_layers=num_layers,
                        batch_first=True,bidirectional=(num_dir==2),
                        dropout=dropout)
    self.dgru = nn.GRU(hidden_size*2, hidden_size*2, num_layers=num_layers, 
                        dropout=dropout)
  def get_hx(self,size):
    hx = autograd.Variable(torch.FloatTensor(self.num_layers*self.num_dir,
                                            size, self.hidden_size).zero_())
    if int(torch.cuda.is_available()) == 1:
        hx.data = hx.data.cuda()
    return hx

  def forward(self, inp):
    he = self.get_hx(inp.size(0))
    enc,h = self.egru(inp,he)
    #enc hidden has bidirection so switch those to the features dim
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) 
    outputs = []
    op = torch.Variable(torch.cuda.FloatTensor(inp.size(0),self.hidden_size*2).zero_(), requires_grad=False)
    for o in outp[1:]: 
      op = op.unsqueeze(0)
      h = self.dgru(op,h)
      print(h.size());exit()
      #attn
     
      



