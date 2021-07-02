import re

import joblib

from keras.models import load_model

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

loaded_model= load_model("model.hdf5")
tfidf_v= joblib.load("tfidf_v.pkl")

stoplist= stopwords.words("english")
lm= WordNetLemmatizer()

def classifier(inp):
  txt= []
  text= re.sub("[^a-zA-Z]", " ",inp)
  text= text.lower()
  text= text.split()
  text= [lm.lemmatize(word) for word in text if word not in stoplist]
  text= " ".join(text)
  txt.append(text)
  x= tfidf_v.transform(txt).toarray()
  y= loaded_model.predict(x)

  return "real" if y >= 0.5 else "fake"
