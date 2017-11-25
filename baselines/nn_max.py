import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from data_multiref import load_data

def calcbleu(generated, targets, end):
  #calcbleu(generated, targets, DS.stoi_targets["<end>"]):
  cc = SmoothingFunction()
  generated = torch.max(generated.data.cpu(),2)[1]
  targets = targets.data.cpu()
  bleu = 0
  for i in range(generated.size(0)):
    gen = list(generated[i])
    targ = list(targets[i])
    genend = gen.index(end) if end in gen else len(gen)
    targend = targ.index(end) if end in targ else len(targ)
    gen = gen[:genend]
    targ = targ[:targend]
    bleu += sentence_bleu([targ],gen,smoothing_function=cc.method3)
  return bleu/i

DS = load_data()


