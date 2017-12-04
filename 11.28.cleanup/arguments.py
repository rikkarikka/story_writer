import os
import argparse

def general():
  parser = argparse.ArgumentParser(description='none')
  # learning
  parser.add_argument('-layers', type=int, default=2, help='min_freq for vocab [default: 1]') #
  parser.add_argument('-drop', type=float, default=0.3, help='min_freq for vocab [default: 1]') #
  parser.add_argument('-bsz', type=int, default=100, help='min_freq for vocab [default: 1]') #
  parser.add_argument('-hsz', type=int, default=500, help='min_freq for vocab [default: 1]') #
  parser.add_argument('-maxlen', type=int, default=75, help='min_freq for vocab [default: 1]') #
  parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]') #
  parser.add_argument('-epochs', type=int, default=30, help='number of epochs for train [default: 100]') #
  parser.add_argument('-debug', action="store_true")
  parser.add_argument('-resume', type=str, default=None)
  parser.add_argument('-train',type=str,default="data/train.txt.ner.9ref")
  parser.add_argument('-valid',type=str,default="data/valid.txt.ner.9ref")
  parser.add_argument('-datafile', type=str, default="data/9ref.pt")
  return parser

def mkdir(args):
  try:
    os.stat(args.savestr)
  except:
    os.mkdir(args.savestr)

def s2s2():
  parser = general()
  parser.add_argument('-savestr',type=str,default="saved_models/9ref_s2s2/")
  args = parser.parse_args()
  args.train = "../data/10refs/t_seq2.txt"
  args.valid = "../data/10refs/v_seq2.txt"
  args.datafile = "data/9ref_seq2.pt"
  mkdir(args)
  return args

def s2s1():
  parser = general()
  parser.add_argument('-savestr',type=str,default="saved_models/9ref_s2s1/")
  args = parser.parse_args()
  args.train = "../data/10refs/t_seq1.txt"
  args.valid = "../data/10refs/v_seq1.txt"
  args.datafile = "data/9ref_seq1.pt"
  mkdir(args)
  return args

def s2s2cats():
  parser = general()
  parser.add_argument('-savestr',type=str,default="saved_models/9ref_cats2s/")
  args = parser.parse_args()
  args.train = "../data/10refs/s2cats2.train"
  args.valid = "../data/10refs/s2cats2.val"
  args.datafile = "data/s2cats2.pt"
  mkdir(args)
  return args
def s2s1cats():
  parser = general()
  parser.add_argument('-savestr',type=str,default="saved_models/9ref_s2cats/")
  args = parser.parse_args()
  args.train = "../data/10refs/s2cats1.train"
  args.valid = "../data/10refs/s2cats1.val"
  args.datafile = "data/s2cats1.pt"
  mkdir(args)
  return args

def s2s2catsrare():
  parser = general()
  parser.add_argument('-savestr',type=str,default="saved_models/9ref_cats2srare/")
  args = parser.parse_args()
  args.train = "../data/10refs/s2rarecats.train"
  args.valid = "../data/10refs/s2rarecats.val"
  args.datafile = "data/s2rarecats.pt"
  mkdir(args)
  return args
def s2s1catsrare():
  parser = general()
  parser.add_argument('-savestr',type=str,default="saved_models/9ref_s2catsrare/")
  args = parser.parse_args()
  args.train = "../data/10refs/s1rarecats.train"
  args.valid = "../data/10refs/s1rarecats.val"
  args.datafile = "data/rarecats2s.pt"
  mkdir(args)
  return args

def s2s2nb():
  parser = general()
  parser.add_argument('-savestr',type=str,default="saved_models/9ref_s2s2nb/")
  args = parser.parse_args()
  args.train = "../data/10refs/t_seq2_nobe.txt"
  args.valid = "../data/10refs/v_seq2_nobe.txt"
  args.datafile = "data/9ref_seq2_nobe.pt"
  mkdir(args)
  return args

def s2s1nb():
  parser = general()
  parser.add_argument('-savestr',type=str,default="saved_models/9ref_s2s1nb/")
  args = parser.parse_args()
  args.train = "../data/10refs/t_seq1_nobe.txt"
  args.valid = "../data/10refs/v_seq1_nobe.txt"
  args.datafile = "data/9ref_seq1_nobe.pt"
  mkdir(args)
  return args

def s2s():
  parser = general()
  parser.add_argument('-savestr',type=str,default="saved_models/9ref_s2s/")
  args = parser.parse_args()
  mkdir(args)
  return args

def noattn_s2s():
  parser = general()
  parser.add_argument('-savestr',type=str,default="saved_models/9ref_noattn_s2s/")
  args = parser.parse_args()
  mkdir(args)
  return args

def sxs():
  parser = general()
  parser.add_argument('-savestr',type=str,default="saved_models/9ref_sxs/")
  args = parser.parse_args()
  mkdir(args)
  return args

def cat_s2s():
  parser = general()
  parser.add_argument('-savestr',type=str,default="saved_models/cat_s2s/")
  parser.add_argument('-datafile',type=str,default="data/cat_data.pt")
  parser.add_argument('-catemb',type=int,default=500)
  parser.add_argument('-lossratio',type=float,default=0.25)
  args = parser.parse_args()
  mkdir(args)
  return args

def s2s1_params():
    parser = argparse.ArgumentParser(description='none')
    # learning
    parser.add_argument('-layers', type=int, default=2, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-drop', type=float, default=0.3, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-bsz', type=int, default=100, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-hsz', type=int, default=500, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-maxlen', type=int, default=75, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]') #
    parser.add_argument('-epochs', type=int, default=30, help='number of epochs for train [default: 100]') #
    parser.add_argument('-datafile', type=str, default="data/seq2seq2seq/s1.pt")
    parser.add_argument('-debug', action="store_true")
    parser.add_argument('-pretrain', action="store_true")
    parser.add_argument('-resume', type=str, default=None)
    parser.add_argument('-eventvocab', type=str, default="data/verbnet_categories.txt")
    parser.add_argument('-temperature', type=float, default=10)
    parser.add_argument('-save', type=str, default="data/seq2seq2seq/s1.pt")
    parser.add_argument('-train', type=str, default="data/seq2seq2seq/train.seq1")
    parser.add_argument('-valid', type=str, default="data/seq2seq2seq/valid.seq1")
    parser.add_argument("-savestr",type=str,default="saved_models/s2s2s/")
 
    args = parser.parse_args()
    return args

def s2s2_params():
    parser = argparse.ArgumentParser(description='none')
    # learning
    parser.add_argument('-layers', type=int, default=2, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-drop', type=float, default=0.3, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-bsz', type=int, default=100, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-hsz', type=int, default=500, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-maxlen', type=int, default=75, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]') #
    parser.add_argument('-epochs', type=int, default=30, help='number of epochs for train [default: 100]') #
    parser.add_argument('-datafile', type=str, default="data/seq2seq2seq/s2.pt")
    parser.add_argument('-debug', action="store_true")
    parser.add_argument('-pretrain', action="store_true")
    parser.add_argument('-resume', type=str, default=None)
    parser.add_argument('-eventvocab', type=str, default="data/verbnet_categories.txt")
    parser.add_argument('-temperature', type=float, default=10)
    parser.add_argument('-save', type=str, default="data/seq2seq2seq/s2.pt")
    parser.add_argument('-train', type=str, default="data/seq2seq2seq/train.seq2")
    parser.add_argument('-valid', type=str, default="data/seq2seq2seq/valid.seq2")
    parser.add_argument("-savestr",type=str,default="saved_models/s2s2s/")
    parser.add_argument("-vcats",type=str,default="data/verb_cats.txt")
    parser.add_argument("-vvocab",type=str,default="data/verb_vocab.txt")
    parser.add_argument("-ndic",type=str,default="data/noundic.pickle")
 
    args = parser.parse_args()
    return args

def s2s2s_eval():
    parser = argparse.ArgumentParser(description='none')

    parser.add_argument('-s1', type=str, default="saved_models/s2s2s/19_bleu-0.0051")
    parser.add_argument('-s2', type=str, default="saved_models/s2s2s/40_bleu-0.0498")
    parser.add_argument('-data', type=str, default="data/multiref.pt")
    parser.add_argument('-s2ds', type=str, default="data/seq2seq2seq/s2.pt")
    parser.add_argument('-s1ds', type=str, default="data/seq2seq2seq/s1.pt")
    parser.add_argument('-savestr', type=str, default="saved_models/s2s2s/")
    parser.add_argument('-bsz', type=int, default=10, help='min_freq for vocab [default: 1]') #

    args = parser.parse_args()
    return args
