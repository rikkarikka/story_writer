import os
import argparse
import torch
from torch import autograd, nn
import torch.nn.functional as F
from numpy import genfromtxt
from torch.autograd import Variable

import model as m
from torchtext import data, datasets
from evalTest import eval,test
from torchtext.vocab import GloVe
from vecHandler import Vecs

def main():
    args = parseParams()
    if not os.path.isdir(args.save_path_full):
        train(args)
    else:
        print('Previously Trained')

def train(args):
    ###############################################################################
    # Load data
    ###############################################################################
    cuda = int(torch.cuda.is_available())-1

    TEXT = data.Field(lower=True,init_token="<start>",eos_token="<end>")
    LABELS = data.Field()#lower=True,init_token="<start>",eos_token="<end>")

    train, val, test = data.TabularDataset.splits(
        path='./', train='train.txt',
        validation='valid.txt', test='test.txt', format='tsv',
        fields=[('text',TEXT),('label',LABELS)])

    TEXT.build_vocab(train)
    LABELS.build_vocab(train)
    vecs = Vecs(args.emb_dim)

    #print('Making interator for splits...')
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.text))#, device=cuda)

    num_classes = len(LABELS.vocab)
    input_size = len(TEXT.vocab)
    ###############################################################################
    # Build the model
    ###############################################################################

    model = m.Model(input_size=input_size, hidden_size=args.hidden_sz,
                    num_classes=num_classes,
                    num_layers=args.num_layers, num_dir=args.num_dir,
                    batch_size=args.batch_size, emb_dim=args.emb_dim,
                    embfix=args.embfix, dropout=args.dropout,
                    net_type=args.net_type)#, device=args.device)
    criterion = nn.CrossEntropyLoss()
    # Select optimizer
    if (args.opt == 'adamax'):
        optimizer = torch.optim.Adamax(model.parameters())#, lr=args.lr)
    elif (args.opt == 'adam'):
        optimizer = torch.optim.Adam(model.parameters())#, lr=args.lr)
    elif (args.opt == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.5)#,lr=args.lr,momentum=0.5)
    else:
        #print('Optimizer unknown, defaulting to adamax')
        optimizer = torch.optim.Adamax(model.parameters())


    ###############################################################################
    # Training the Model
    ###############################################################################
    if cuda == 0:
        model = model.cuda()#args.device)

    highest_t1_acc = 0
    highest_t1_acc_metrics = ''
    highest_t1_acc_params = ''
    results = ''
    for epoch in range(args.epochs):
        losses = []
        tot_loss = 0
        train_iter.repeat=False
        for batch_count,batch in enumerate(train_iter):
            model.zero_grad()
            inp = batch.text.t()
            inp3d = torch.cuda.FloatTensor(inp.size(0),inp.size(1),args.emb_dim)
            for i in range(inp.size(0)):
              for j in range(inp.size(1)):
                inp3d[i,j,:] = vecs[TEXT.vocab.itos[inp[i,j].data[0]]]
            outp = batch.label.t()
            print("OUTP: ",outp.size())
            outp3d = torch.cuda.FloatTensor(outp.size(0),outp.size(1),args.emb_dim)
            for i in range(outp.size(0)):
              for j in range(outp.size(1)):
                outp3d[i,j,:] = vecs[LABELS.vocab.itos[outp[i,j].data[0]]]
            print("INP: ",inp.size())

            preds, attns = model(Variable(inp3d),Variable(outp3d,requires_grad=False))
            print("PREDS: ",preds.size())
            preds = preds.view(-1,preds.size(2))
            #print("LABELS: ",batch.label.size())
            loss = criterion(preds, batch.label.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss)
            tot_loss += loss.data[0]

            #if (batch_count % 20 == 0):
            #    print('Batch: ', batch_count, '\tLoss: ', str(losses[-1].data[0]))
        #print('Average loss over epoch ' + str(epoch) + ': ' + str(tot_loss/len(losses)))
        (avg_loss, accuracy, corrects, size, t5_acc, t5_corrects, mrr) = eval(val_iter, model,vecs,TEXT,args.emb_dim)#, args.device)
        if accuracy > args.acc_thresh:
            save_path = '{}/acc{:.2f}_e{}.pt'.format(args.save_path_full, accuracy, epoch)
            if not os.path.isdir(args.save_path_full):
                os.makedirs(args.save_path_full)
            torch.save(model, save_path)

        if highest_t1_acc < accuracy:
            highest_t1_acc = accuracy
            highest_t1_acc_metrics = ('acc: {:6.4f}%({:3d}/{}) EPOCH{:2d} - loss: {:.4f} t5_acc: {:6.4f}%({:3d}' \
                    '/{}) MRR: {:.6f}'.format(accuracy, corrects, size,epoch, avg_loss, t5_acc, t5_corrects, size, mrr))

            highest_t1_acc_params = (('PARAMETERS:' \
                    'net-%s' \
                    '_e%i' \
                    '_bs%i' \
                    '_opt-%s' \
                    '_ly%i' \
                    '_hs%i' \
                    '_dr%i'
                    '_ed%i' \
                    '_femb%s' \
                    '_ptemb%s' \
                    '_drp%.1f' \
                    '_mf%d\n'
                    % (args.net_type, args.epochs, args.batch_size, args.opt, args.num_layers,
                    args.hidden_sz, args.num_dir, args.emb_dim, args.embfix, args.pretr_emb, args.dropout, args.mf)))
        results += ('\nEPOCH{:2d} - loss: {:.4f}  acc: {:6.4f}%({:3d}/{}) t5_acc: {:6.4f}%({:3d}' \
                '/{}) MRR: {:.6f}'.format(epoch, avg_loss, accuracy,
                                        corrects, size, t5_acc, t5_corrects, size,
                                        mrr))

    print(highest_t1_acc_metrics + '\n')
    writeResults(args, results, highest_t1_acc, highest_t1_acc_metrics, highest_t1_acc_params)

def writeResults(args, results, highest_t1_acc, highest_t1_acc_metrics, highest_t1_acc_params):
    if not os.path.isdir(args.save_path_full):
        os.makedirs(args.save_path_full)
    f = open(args.save_path_full + '/results.txt','w')
    f.write('PARAMETERS:\n' \
            'Net Type: %s\n' \
            #'Learning Rate: %f\n' \
            'Epochs: %i\n' \
            'Batch Size: %i\n' \
            'Optimizer: %s\n' \
            'Num Layers: %i\n' \
            'Hidden Size: %i\n' \
            'Num Directions: %i\n'
            'Embedding Dimension: %i\n' \
            'Fixed Embeddings: %s\n' \
            'Pretrained Embeddings: %s\n' \
            'Dropout: %.1f\n' \
            'Min Freq: %d'
            % (args.net_type, args.epochs, args.batch_size, args.opt, args.num_layers,
            args.hidden_sz, args.num_dir, args.emb_dim, args.embfix, args.pretr_emb, args.dropout, args.mf))
    f.write(results)
    f.close()
    if highest_t1_acc > args.acc_thresh:
        g = open(args.save_path + args.folder+ '/best_models.txt','a')
        g.write(highest_t1_acc_metrics)
        g.write(highest_t1_acc_params)
        g.close()

def parseParams():
    parser = argparse.ArgumentParser(description='LSTM text classifier')
    # learning
    parser.add_argument('-mf', type=int, default=1, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]') #
    parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 100]') #
    parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]') #
    parser.add_argument('-opt', type=str, default='adamax', help='optimizer [default: adamax]') #

    # model
    parser.add_argument('-net-type', type=str, default='lstm', help='network type [default: lstm]')
    parser.add_argument('-num-layers', type=int, default=4, help='number of layers [default: 1]') #
    parser.add_argument('-hidden-sz', type=int, default=500, help='hidden size [default: 300]') #
    parser.add_argument('-num-dir', type=int, default=2, help='number of directions [default: 2]') #
    parser.add_argument('-emb-dim', type=int, default=300, help='number of embedding dimension [default: 300]') #
    parser.add_argument('-embfix', type=str, default=False, help='fix the embeddings [default: False]') #
    parser.add_argument('-pretr-emb', type=str, default=False, help='use pretrained embeddings') #
    parser.add_argument('-dropout', type=float, default=.5, help='dropout rate [default: .5]')

    # options
    parser.add_argument('-save-path', type=str, default='./saved_models', help='path to save models [default: ./saved_models]')
    parser.add_argument('-folder', type=str, default='', help='folder to save models [default: '']')
    parser.add_argument('-acc-thresh', type=float, default=40, help='top1 accuracy threshold to save model')
    parser.add_argument('-device', type=int, default=1, help='GPU to use [default: 1]')
    args = parser.parse_args()

    args.embfix = (args.embfix == 'True')
    args.pretr_emb = (args.pretr_emb == 'True')

    args.save_path_full = args.save_path + \
                        args.folder + \
                        '/net-' + str(args.net_type) + \
                        '_e' + str(args.epochs) + \
                        '_bs' + str(args.batch_size) + \
                        '_opt-' + str(args.opt) + \
                        '_ly' + str(args.num_layers) + \
                        '_hs' + str(args.hidden_sz) + \
                        '_dr' + str(args.num_dir) + \
                        '_ed' + str(args.emb_dim) + \
                        '_femb' + str(args.embfix) + \
                        '_ptemb' + str(args.pretr_emb) + \
                        '_drp' + str(args.dropout)
    if args.mf > 1: args.save_path_full += '_mf' + str(args.mf)
    return args

if __name__ == '__main__':
    main()

