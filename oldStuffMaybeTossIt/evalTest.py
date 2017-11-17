import torch
from torch import autograd, nn
from torch.autograd import Variable
import torch.nn.functional as F

def eval(data_iter, model,vecs,TEXT,LABELS,loss,emb_dim):
    model.eval()
    corrects, avg_loss, t5_corrects, rr = 0, 0, 0, 0
    for batch_count,batch in enumerate(data_iter):
        #print('avg_loss:', avg_loss)
        inp, target = batch.text, batch.label
        inp.data.t_()#, target.data.sub_(1)  # batch first, index align
        inp3d = torch.cuda.FloatTensor(inp.size(0),inp.size(1),emb_dim)
        for i in range(inp.size(0)):
          for j in range(inp.size(1)):
            inp3d[i,j,:] = vecs[TEXT.vocab.itos[inp[i,j].data[0]]]
        #if args.cuda:
        #    feature, target = feature.cuda(), target.cuda()

        outp = batch.label.t()
        outp3d = torch.cuda.FloatTensor(outp.size(0),outp.size(1),emb_dim)
        for i in range(outp.size(0)):
          for j in range(outp.size(1)):
            outp3d[i,j,:] = vecs[LABELS.vocab.itos[outp[i,j].data[0]]]

        preds, attns = model(Variable(inp3d),Variable(outp3d,requires_grad=False))
        loss,grad,numcorrect = loss(preds, batch.label, model.generate,criterion,eval=True)

        avg_loss += loss

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    model.train()
    print("EVAL: ",avg_loss)

    return avg_loss#, accuracy, corrects, size, t5_acc, t5_corrects, mrr);

def test(text, model, text_field, label_field):
    model.eval()
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return predicted.data[0][0]+1
