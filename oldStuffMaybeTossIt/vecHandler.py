import torchtext
import torch

class Vecs:
  def __init__(self,emb_dim):
    self.gl = torchtext.vocab.GloVe(name='6B', dim=emb_dim, unk_init=torch.FloatTensor.uniform_)
    self.cache = {}

  def __getitem__(self, w):
    try:
      x = self.cache[w]
    except:
      x = self.gl[w]
      x = x.squeeze()
      self.cache[w] = x
    return x


if __name__=="__main__":
  v = Vecs(300)
  print(v['<person>'])
  print(v['<persoaan>'])
