from preprocess import *
from arguments import s2s1catsrare as parseParams
if __name__=="__main__":
  args = parseParams()
  DS = load_data(args)
  torch.save(DS,args.datafile)
