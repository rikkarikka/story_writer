import sys
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from arguments import s2s_inter as parseParams
from inter_joint import surface
from inter_preprocess_new import load_data
import pickle
  
def draw(inters,surface,attns,args):
  for i in range(len(inters)):
    try:
      os.mkdir(args.savestr+"attns/")
    except:
      pass
    with open(args.savestr+"attns/"+args.epoch+"-"+str(i),'wb') as f:
      pickle.dump((inters[i],surface[i],attns[i].data.cpu().numpy()),f)

def validate(S,DS,args,m):
  print(m,args.valid)
  data = DS.new_data(args.valid)
  cc = SmoothingFunction()
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
      l,a = S.beamsearch(s)
      logits.append(l)
      attn.append(a)
      #ihyp = [[DS.verb_vocab[x] for x in list(y)] for y in preds.data]
      #inters.extend(ihyp)
    attns.append(torch.cat(a,0))
    logits = logits[0]
    hyp = [DS.vocab[x] for x in logits]
    hyps.append(hyp)
    print(hyp)
    refs.append(targets)
    assert(len(hyps)==len(refs))
  draw(inters,hyps,attns,args)
  bleu = corpus_bleu(refs,hyps,emulate_multibleu=True,smoothing_function=cc.method3)
  print(bleu)
  S.train()
  with open(args.savestr+"hyps"+m+"-bleu_"+str(bleu),'w') as f:
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

def main(args,m):
  DS = torch.load(args.datafile)
  DS.args = args
  print(m)
  args.epoch = m
  S,_ = torch.load(args.savestr+m)
  if not args.cuda:
    print('move to cpu')
    S = S.cpu()
  S.dec.flatten_parameters()
  S.enc.flatten_parameters()
  S.vdec.flatten_parameters()
  S.args = args
  S.endtok = DS.vocab.index("<eos>")
  S.vendtok = DS.verb_vocab.index("<eos>")
  S.punct = [DS.vocab.index(t) for t in ['.','!','?']]
  validate(S,DS,args,m)

if __name__=="__main__":
  args = parseParams()
  if args.vmodel:
    models = [args.vmodel]
  else:
    models = [x for x in os.listdir(args.savestr) if x[0].isdigit()]
    models.sort(reverse=True)
  for m in models:
    main(args,m)
