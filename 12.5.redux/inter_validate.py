
  
def draw(inters,surface,attns,args):
  for i in range(len(inters)):
    try:
      os.mkdir(args.savestr+"attns/")
    except:
      pass
    with open(args.savestr+"attns/"+args.epoch+"-"+str(i),'wb') as f:
      pickle.dump((inters[i],surface[i],attns[i].data.cpu().numpy()),f)

def validate(I,S,DS,args):
  data = DS.val_batches
  cc = SmoothingFunction()
  I.eval()
  S.eval()
  refs = []
  hyps = []
  attns = []
  inters = []
  for x in data:
    if args.debug:
      sources, targets, _= DS.pad_batch(x,targ=False)
    else:
      sources, targets = DS.pad_batch(x,targ=False,v=False)
    sources = Variable(sources,requires_grad=False)
    logits = []
    attn = []
    for s in sources:
      s = s.unsqueeze(0).contiguous()
      predinter,(he,ce) = I.forward(s)
      _, preds = torch.max(predinter,2)
      plist = list(preds.data[0])
      if 1 in plist[:-1]:
        preds = preds[:,:plist.index(1)+1]
      p,(hv,cv) = I.encode(preds.data,(he,ce))
      h = hv; c = cv
      l,a = S.beamsearch(p,(h,c))
      logits.append(l)
      attn.append(a)
      ihyp = [[DS.verb_vocab[x] for x in list(y)] for y in preds.data]
      inters.extend(ihyp)
    attns.append(torch.cat(a,0))
    hyp = [x[:x.index(1)] if 1 in x else x for x in logits]
    hyp = [[DS.vocab[x] for x in y] for y in hyp]
    hyps.extend(hyp)
    refs.extend(targets)
  draw(inters,hyps,attns,args)
  bleu = corpus_bleu(refs,hyps,emulate_multibleu=True,smoothing_function=cc.method3)
  I.train()
  S.train()
  with open(args.savestr+"hyps"+args.epoch,'w') as f:
    hyps = [' '.join(x) for x in hyps]
    f.write('\n'.join(hyps))
  try:
    os.stat(args.savestr+"refs")
  except:
    with open(args.savestr+"refs",'w') as f:
      refstr = []
      for r in refs:
        r = [' '.join(x) for x in r]
        refstr.append('\n'.join(r))
      f.write('\n'.join(refstr))
  return bleu

