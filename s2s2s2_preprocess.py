import sys
import torch

from s1_preprocess import load_data,vecs
s1ds = torch.load("data/seq2seq2seq/s1.pt")

from s2_preprocess import load_data,vecs
s2ds = torch.load("data/seq2seq2seq/s2.pt")

s1ds.mkbatches(64)
s2ds.mkbatches(64)

