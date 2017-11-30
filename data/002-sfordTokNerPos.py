import sys
from pynlp import StanfordCoreNLP

annotators = 'tokenize, ssplit, pos, lemma, ner'
nlp = StanfordCoreNLP(annotators=annotators)

titles = open("titles.tok")
nstories = []
ntitles = []
nvs = []
ctr = 0
with open("stories.all") as f:
    for l in f:
        print(ctr)
        ctr+=1
        #if ctr>10: break
        title = next(titles)
        annot = nlp(l)
        newl = []
        people = []
        locs = []
        nv = []
        for s in annot:
            for t in s:
                if t.ner == "PERSON":
                    newl.append(t.ner)
                    people.append(t.word)
                elif t.ner == "LOCATION":
                    newl.append(t.ner)
                    locs.append(t.word)
                else:
                    newl.append(t.word)
                    if t.pos[0] in ['N','V']:
                        nv.append(t.word+"_"+t.pos)
        nstories.append(" ".join(newl))
        nvs.append(" ".join(nv))
        for w in people:
            w = w.lower()
            if w in title:
                title = title.replace(w,"PERSON")
        for w in locs:
            w = w.lower()
            if w in title:
                title = title.replace(w,"LOCATION")
        ntitles.append(title)
titles.close()
with open("stories.all.ner",'w') as f:
    f.write("\n".join(nstories))
with open("titles.all.ner",'w') as f:
    f.write("".join(ntitles))
with open("nv.all",'w') as f:
    f.write("\n".join(nvs))
