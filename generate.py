import torch
from torch.autograd import Variable
import argparse
from data import load_data

def genmax(logits,vocab,targets=None):
  logits = torch.max(logits.data.cpu(),2)[1]
  texts = []
  targs = []
  for i,l in enumerate(logits):
    wrds = [vocab[k] for k in l]
    end = wrds.index("<end>") if "<end>" in wrds else len(wrds)
    txt = " ".join(wrds[:end])
    texts.append(txt)
    if targets is not None:
      l = targets[i].data.cpu()
      wrds = [vocab[k] for k in l]
      end = wrds.index("<end>") if "<end>" in wrds else len(wrds)
      txt = " ".join(wrds[:end])
      targs.append(txt)

  return texts,targs


def validate(M,DS,args):
  M.eval()
  data = DS.val_batches
  gens = []
  targs = []
  for sources,targets in data:
    sources = Variable(sources.cuda(),volatile=True)
    targets = Variable(targets.cuda(),volatile=True)
    M.zero_grad()
    outs = M(sources)
    logits = M.gen(outs)
    targets = targets[:,1:].contiguous()
    gen, targ = genmax(logits,DS.itos_targets,targets=targets)
    gens.extend(gen)
    targs.extend(targ)
  return gens,targs


def main():
  args = parseParams()
  DS = torch.load(args.datafile)
  args.vsz = len(DS.itos_targets)
  args.start = DS.stoi_targets["<start>"]
  M,_ = torch.load(args.model)
  gen, targ = validate(M,DS,args)
  with open(args.model+".gen",'w') as f:
    f.write("\n".join(gen))
  with open(args.model+".targ",'w') as f:
    f.write("\n".join(targ))

def parseParams():
    parser = argparse.ArgumentParser(description='none')
    # learning
    parser.add_argument('-layers', type=int, default=2, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-bsz', type=int, default=50, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-hsz', type=int, default=500, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-maxlen', type=int, default=75, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-model', type=str, required=True)
    parser.add_argument('-datafile', type=str, default="data/datafile.pt")

 
    args = parser.parse_args()
    return args

if __name__=="__main__":
  main()
