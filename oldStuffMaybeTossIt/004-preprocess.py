import sys
import torch
import torchtext

def data(args):
    ###############################################################################
    # Load data
    ###############################################################################
    cuda = int(torch.cuda.is_available())-1

    SRC = data.Field(lower=True,init_token="<start>",eos_token="<end>")
    TGT = data.Field(lower=True,init_token="<start>",eos_token="<end>")

    train, val, test = data.TabularDataset.splits(
        # Kushman data
        path='./', train='train.txt',
        validation='valid.txt', test='test.txt', format='tsv',
        fields=[('src',SRC), ('tgt', TGT)])

    SRC.build_vocab(train,vectors=GloVe(name='6B', dim=args.emb_dim),min_freq=args.mf)
    prevecs=SRC.vocab.vectors
    TGT.build_vocab(train)

    #print('Making interator for splits...')
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.src))

    input_size = len(SRC.vocab)
