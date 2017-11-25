import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import defaultdict
from itertools import product

class CategoricalLoss(nn.Module):

  def __init__(self,stoi,args):
    super(CategoricalLoss,self).__init__()
    with open("data/vocabulary.txt") as f:
      ctr = 0
      wordassoc = defaultdict(list)
      for l in f:
        l = l.strip()
        if l == "___":
          ctr+=1
        else:
          wordassoc[l].append(ctr)
    self.idxassoc = {stoi[k]:torch.cuda.LongTensor(v) for k,v in wordassoc.items() if k in stoi}
    weights = torch.cuda.FloatTensor(len(stoi)).fill_(1)
    weights[0] = 0
    self.criterion = nn.CrossEntropyLoss(weights)
    self.args = args

  def forward(self, logits, targets, attns):
    _, gens = torch.max(logits,2)
    attnloss = Variable(torch.cuda.FloatTensor(gens.size()).zero_())
    for i,j in product(range(attnloss.size(0)),range(attnloss.size(1))):
      if gens[i,j].data[0] in self.idxassoc:
        loss = (1- attns[i,:,j][self.idxassoc[gens[i,j].data[0]]].sum())**2
        attnloss[i,j] = loss

    attnloss = attnloss.view(-1).mean()
    logits = logits[:,:targets.size(1),:].contiguous()
    logits = logits.view(-1,logits.size(2))
    #if targets.size(1)>self.args.maxlen:
    #  targets = targets[:,:self.args.maxlen].contiguous()
    targets = targets.view(-1)
    xent = self.criterion(logits,targets)
    ret = xent+attnloss
    return ret

if __name__=='__main__':
  c = CategoricalLoss(None,None)
