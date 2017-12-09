import sys
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from arguments import s2s as parseParams
from preprocess_new import load_data
import pickle
  
def draw(inters,surface,attns,args):
  for i in range(len(inters)):
    try:
      os.mkdir(args.savestr+"attns/")
    except:
      pass
    with open(args.savestr+"attns/"+args.epoch+"-"+str(i),'wb') as f:
      pickle.dump((inters[i],surface[i],attns[i].data.cpu().numpy()),f)

def validate(I,S,DS,args,m):
  print(m,args.valid)
  data = DS.new_data(args.valid)
  cc = SmoothingFunction()
  I.eval()
  S.eval()
  refs = []
  hyps = []
  attns = []
  inters = []
  for sources,targets in data:
    sources = Variable(sources,requires_grad=False)
    logits = []
    attn = []
    for s in sources:
      s = s.unsqueeze(0).contiguous()
      predinter,(he,ce) = I.forward(s)
      _, preds = torch.max(predinter,2)
      plist = list(preds.data[0])
      if 1 in plist[:-1]:
        preds = preds[:,:plist.index(1)+1]
      p,(hv,cv) = I.encode(preds.data,(he,ce))
      h = hv; c = cv
      l,a = S.beamsearch(p,(h,c))
      logits.append(l)
      attn.append(a)
      ihyp = [[DS.verb_vocab[x] for x in list(y)] for y in preds.data]
      inters.extend(ihyp)
    attns.append(torch.cat(a,0))
    logits = logits[0]
    hyp = [DS.vocab[x] for x in logits]
    hyps.append(hyp)
    refs.append(targets)
    assert(len(hyps)==len(refs))
  draw(inters,hyps,attns,args)
  bleu = corpus_bleu(refs,hyps,emulate_multibleu=True,smoothing_function=cc.method3)
  I.train()
  S.train()
  with open(args.savestr+"hyps"+m,'w') as f:
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

def main():
  args = parseParams()
  DS = torch.load(args.datafile)
  DS.args = args
  models = [x for x in os.listdir(args.savestr) if x[0].isdigit()]
  for m in models:
    args.epoch = m
    I,S,_,_ = torch.load(args.savestr+m)
    if not args.cuda:
      print('move to cpu')
      S = S.cpu()
      I = I.cpu()
    S.dec.flatten_parameters()
    I.enc.flatten_parameters()
    I.vdec.flatten_parameters()
    S.args = args
    I.args = args
    S.endtok = DS.vocab.index("<eos>")
    S.punct = [DS.vocab.index(t) for t in ['.','!','?']]
    validate(I,S,DS,args,m)

if __name__=="__main__":
  main()
