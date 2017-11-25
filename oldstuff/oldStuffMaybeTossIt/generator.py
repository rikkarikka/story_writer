import sys
import os
import argparse
import torch
from torch import autograd, nn
import torch.nn.functional as F
from numpy import genfromtxt
from torch.autograd import Variable

import model as m
from torchtext import data, datasets
#from evalTest import eval,test
from torchtext.vocab import GloVe
from vecHandler import Vecs

def main():
    args = parseParams()
    if not os.path.isdir(args.save_path_full):
        train(args)
    else:
        print('Previously Trained')

def generate(data_iter, model, vecs, TEXT, LABELS, emb_dim, fn="lastPredictions.txt"):
  model.eval()
  f = open(fn,'w') 
  corrects, avg_loss, t5_corrects, rr = 0, 0, 0, 0
  for batch_count,batch in enumerate(data_iter):
    print(batch_count)
    model.zero_grad()
    #print('avg_loss:', avg_loss)
    inp, target = batch.text, batch.label
    inp.data.t_()#, target.data.sub_(1)  # batch first, index align
    inp3d = torch.cuda.FloatTensor(inp.size(0),inp.size(1),emb_dim)
    for i in range(inp.size(0)):
      for j in range(inp.size(1)):
        inp3d[i,j,:] = vecs[TEXT.vocab.itos[inp[i,j].data[0]]]
    #if args.cuda:
    #    feature, target = feature.cuda(), target.cuda()

    outp = batch.label.t()
    outp3d = torch.cuda.FloatTensor(outp.size(0),outp.size(1),emb_dim)
    for i in range(outp.size(0)):
      for j in range(outp.size(1)):
        outp3d[i,j,:] = vecs[LABELS.vocab.itos[outp[i,j].data[0]]]

    out_t, attns = model(Variable(inp3d),Variable(outp3d,requires_grad=False))
    scores_t = model.generate(out_t)
    pred_t = scores_t.max(2)[1].t().data
    for p in range(len(pred_t)):
      pred = [LABELS.vocab.itos[k] for k in pred_t[p]]
      if "<end>" in pred: pred = pred[:pred.index("<end>")]
      predtxt = " ".join(pred)
      intxt = " ".join([TEXT.vocab.itos[k] for k in inp.data[p]])
      f.write(intxt + " " + predtxt +'\n')
      print(intxt + " " + predtxt +'\n')
  f.close()

if __name__=="__main__":
  model = torch.load(sys.argv[1])
  model.egru.flatten_parameters()
  model.dgru.flatten_parameters()

  TEXT = data.Field(lower=True,init_token="<start>",eos_token="<end>")
  LABELS = data.Field()#lower=True,init_token="<start>",eos_token="<end>")

  train, val, test = data.TabularDataset.splits(
      path='./', train='train.txt',
      validation='valid.txt', test='test.txt', format='tsv',
      fields=[('text',TEXT),('label',LABELS)])

  TEXT.build_vocab(train)
  LABELS.build_vocab(train)
  vecs = Vecs(300)

  train_iter, val_iter, test_iter = data.BucketIterator.splits(
      (train, val, test), batch_sizes=(2,2,2),
      sort_key=lambda x: len(x.text))#, device=0)

  generate(val_iter, model, vecs, TEXT, LABELS, 300)
