import sys
import torch
from torch.autograd import Variable
from s2s import model
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from preprocess import load_data

def validate(M,DS,savestr,data=None):
  if not data:
    data = DS.val_batches
  cc = SmoothingFunction()
  M.eval()
  refs = []
  hyps = []
  for x in data:
    sources, targets = DS.pad_batch(x,targ=False)
    sources = Variable(sources.cuda(),volatile=True)
    M.zero_grad()
    logits = M(sources,None)
    logits = torch.max(logits.data.cpu(),2)[1]
    logits = [list(x) for x in logits]
    hyp = [x[:x.index(1)+1] if 1 in x else x for x in logits]
    hyp = [[DS.vocab[x] for x in y] for y in hyp]
    hyps.extend(hyp)
    refs.extend(targets)
  bleu = corpus_bleu(refs,hyps,emulate_multibleu=True,smoothing_function=cc.method3)
  M.train()
  with open(savestr+"hyps",'w') as f:
    hyps = [' '.join(x) for x in hyps]
    f.write('\n'.join(hyps))
  with open(savestr+"refs",'w') as f:
    refstr = []
    for r in refs:
      r = [' '.join(x) for x in r]
      refstr.append('\n'.join(r))
    f.write('\n'.join(refstr))
  return bleu
'''
def validate(M,DS):
  data = DS.val_batches
  cc = SmoothingFunction()
  M.eval()
  refs = []
  hyps = []
  for x in data:
    sources, targets = DS.pad_batch(x,targ=False)
    sources = Variable(sources.cuda(),volatile=True)
    M.zero_grad()
    logits = M(sources,None)
    logits = torch.max(logits.data.cpu(),2)[1]
    logits = [list(x) for x in logits]
    hyp = [x[:x.index(1)+1] if 1 in x else x for x in logits]
    hyp = [[DS.vocab[x] for x in y] for y in hyp]
    hyps.extend(hyp)
    refs.extend(targets)
  print(len(refs),len(hyps))
  bleu = corpus_bleu(refs,hyps,emulate_multibleu=True,smoothing_function=cc.method3)
  M.train()
  with open("debugging_refs",'w') as f:
    refstr = []
    for r in refs:
      r = [' '.join(x) for x in r]
      refstr.append('\n'.join(r))
    f.write('\n'.join(refstr))
  return bleu
'''
if __name__=="__main__":
  M,_ = torch.load(sys.argv[1] )
  DS = torch.load(sys.argv[2])
  try:
    if sys.argv[3]=='train':
      data = DS.train_batches
      savestr = "train"
    else:
      assert(False)
  except:
    savestr = "val"
    data = DS.val_batches
  savestr += "_" + ''.join([x for x in sys.argv[1] if x!='/'])
  print(validate(M,DS,savestr,data))
