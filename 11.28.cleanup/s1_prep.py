from preprocess import *
from arguments import s2s1 as parseParams
if __name__=="__main__":
  args = parseParams()
  DS = load_data(args)
  torch.save(DS,args.datafile)
