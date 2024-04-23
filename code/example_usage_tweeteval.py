# Most of this code is derived from https://github.com/cardiffnlp/tweeteval/blob/main/TweetEval_Tutorial.ipynb

import transformers
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, TFAutoModel, Pipeline, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
import numpy as np
from scipy.spatial.distance import cosine
from scipy.special import softmax
from collections import defaultdict
import csv
import urllib.request

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

################### Tweet Similarity Scoring #######################

MODEL = "cardiffnlp/twitter-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)

def get_embedding(text):
  text = preprocess(text)
  encoded_input = tokenizer(text, return_tensors='pt')
  features = model(**encoded_input)
  features = features[0].detach().cpu().numpy() 
  features_mean = np.mean(features[0], axis=0) 
  return features_mean

query = "The book was awesome"

tweets = ["I just ordered fried chicken 🐣", 
          "The movie was great", 
          "What time is the next game?", 
          "Just finished reading 'Embeddings in NLP'"]

d = defaultdict(int)
for tweet in tweets:
  sim = 1-cosine(get_embedding(query),get_embedding(tweet))
  d[tweet] = sim

print('Most similar to: ',query)
print('----------------------------------------')
for idx,x in enumerate(sorted(d.items(), key=lambda x:x[1], reverse=True)):
  print(idx+1,x[0])

################### Feature Extraction ######################
  
MODEL = "cardiffnlp/twitter-roberta-base"
text = "Good night 😊"
text = preprocess(text)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Pytorch
encoded_input = tokenizer(text, return_tensors='pt')
model = AutoModel.from_pretrained(MODEL)
features = model(**encoded_input)
features = features[0].detach().cpu().numpy() 
features_mean = np.mean(features[0], axis=0) 
#features_max = np.max(features[0], axis=0)

# # Tensorflow
# encoded_input = tokenizer(text, return_tensors='tf')
# model = TFAutoModel.from_pretrained(MODEL)
# features = model(encoded_input)
# features = features[0].numpy()
# features_mean = np.mean(features[0], axis=0) 
# #features_max = np.max(features[0], axis=0)

print(features_mean.shape)

################### Sentiment Classification ######################

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# load label mapping
mapping_link = "https://raw.githubusercontent.com/DhyeyMavani2003/DeepSentiment/main/data/tweeteval/sentiment/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]
print(labels)

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL) # try running the following if it throws an error: /Applications/Python\ 3.9/Install\ Certificates.command   

text = "Good night 😊"
text = preprocess(text)
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)

# # TF
"""
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

text = "Good night 😊"
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
scores = output[0][0].numpy()
scores = softmax(scores)
"""

ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = labels[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 3)}")