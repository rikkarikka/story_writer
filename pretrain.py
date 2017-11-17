import sys
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from predata import load_data


class model(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.args = args
    self.enc = nn.LSTM(300,args.hsz//2,bidirectional=True,num_layers=args.layers,batch_first=True)
    self.enclin = nn.Linear(args.hsz*args.layers,300)
    self.sm = nn.Softmax()

  def mkevents(self,em):
    self.em = Variable(em.unsqueeze(0).expand(self.args.bsz,em.size(0),em.size(1)))

  def forward(self,inp,out=None):
    enc,(h,c) = self.enc(inp)
    h = h.permute(1,0,2).contiguous()
    h = h.view(h.size(0),-1)
    encdown = self.enclin(h).unsqueeze(1)

    #calculate event attentions
    eW = torch.bmm(encdown,self.em.permute(0,2,1)).squeeze()
    #eW = self.sm(eW)

    return eW

def init_vocab(M,DS,vocab):
  matrix = DS.matrix(vocab).cuda()
  M.mkevents(matrix)

def parseParams():
  parser = argparse.ArgumentParser(description='none')
  parser.add_argument('-layers', type=int, default=2, help='min_freq for vocab [default: 1]') 
  parser.add_argument('-drop', type=float, default=0.3, help='min_freq for vocab [default: 1]') 
  parser.add_argument('-bsz', type=int, default=50, help='min_freq for vocab [default: 1]') 
  parser.add_argument('-hsz', type=int, default=500, help='min_freq for vocab [default: 1]')
  parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]') 
  parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 100]') 
  args = parser.parse_args()
  return args
 
def train(M,DS,optimizer): 
  losses = []
  criterion = nn.MSELoss()
  for source, target, _ in DS.train_batches:
    M.zero_grad()
    preds = M(Variable(source).cuda())
    loss = criterion(preds,Variable(target).cuda())
    loss.backward()
    optimizer.step()
    losses.append(loss.data)
  return sum(losses)/len(losses)
    
def valid(M,DS,epoch,best=1):
  M.eval()
  losses = []
  criterion = nn.MSELoss()
  for source, target, _ in DS.val_batches:
    preds = M(Variable(source).cuda())
    loss = criterion(preds.view(-1),Variable(target).cuda().view(-1))
    losses.append(loss.data)
    out = torch.stack([torch.mul(preds.data,target.cuda()).sum(1),target.cuda().sum(1)],1)
  losses = sum(losses)/len(losses)
  print("Val loss ",losses[0])
  if losses[0] < best:
    best = losses[0]
    torch.save(M,"pretrained/best.state_dict")
  M.train()
  return best
  
def main():
  best = 1
  args = parseParams()
  with open("data/categories.txt") as f:
    cats = f.read().strip().split('\n')
  DS = load_data(50,cats,train="data/train.txt.phrases.pretrain",valid="data/valid.txt.phrases.pretrain")
  M = model(args)
  M = M.cuda()
  init_vocab(M,DS,cats)
  optimizer = torch.optim.Adam(M.parameters(), lr=args.lr)
  for e in range(args.epochs):
    trainloss = train(M,DS,optimizer)
    print(trainloss[0])
    best = valid(M,DS,e,best)

main()
