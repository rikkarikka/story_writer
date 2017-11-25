import argparse

def parseParams():
    parser = argparse.ArgumentParser(description='none')
    # learning
    parser.add_argument('-layers', type=int, default=2, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-drop', type=float, default=0.3, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-bsz', type=int, default=100, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-hsz', type=int, default=500, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-maxlen', type=int, default=75, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]') #
    parser.add_argument('-epochs', type=int, default=30, help='number of epochs for train [default: 100]') #
    parser.add_argument('-datafile', type=str, default="data/multiref.pt")
    parser.add_argument('-debug', action="store_true")
    parser.add_argument('-pretrain', action="store_true")
    parser.add_argument('-resume', type=str, default=None)
    parser.add_argument('-eventvocab', type=str, default="data/verbnet_categories.txt")
    parser.add_argument('-temperature', type=float, default=10)
 
    args = parser.parse_args()
    return args

