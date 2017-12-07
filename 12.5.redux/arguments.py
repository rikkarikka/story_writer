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
  return parser

def mkdir(args):
  try:
    os.stat(args.savestr)
  except:
    os.mkdir(args.savestr)

def s2s_bland():
  parser = general()
  parser.add_argument('-datafile', type=str, default="data/9ref.pt")
  parser.add_argument('-savestr',type=str,default="saved_models/9ref_s2s_bland/")
  parser.add_argument('-beamsize', type=int, default=4, help='min_freq for vocab [default: 1]') #
  args = parser.parse_args()
  mkdir(args)
  return args

def s2s():
  parser = general()
  parser.add_argument('-datafile', type=str, default="data/9ref_verbs.pt")
  parser.add_argument('-savestr',type=str,default="saved_models/9ref_s2s_verbs/")
  parser.add_argument('-verbs',type=str,default="data/train.verbs")
  parser.add_argument('-vmaxlen', type=int, default=10, help='min_freq for vocab [default: 1]') #
  parser.add_argument('-pretrainepochs', type=int, default=10, help='min_freq for vocab [default: 1]') #
  parser.add_argument('-beamsize', type=int, default=4, help='min_freq for vocab [default: 1]') #
  args = parser.parse_args()
  mkdir(args)
  return args
