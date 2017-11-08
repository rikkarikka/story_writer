import sys
import os
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def get_data():
  with open("src-train.txt") as f:
    titles = f.read().strip().split("\n")
  with open("tgt-train.txt") as f:
    stories = f.read().strip().split("\n")
  return titles,stories

titles,stories = get_data()
m = TfidfVectorizer(max_df=0.99,stop_words='english')
rows = m.fit_transform(stories)
pickle.dump((m,rows,titles),open("stories.tfidf",'wb'))
movied = {k:v for k,v in enumerate(titles)}
pickle.dump(movied,open("indexes.pkl",'wb'))

