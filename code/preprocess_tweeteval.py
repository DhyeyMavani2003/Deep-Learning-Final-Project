import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
import warnings
import os
warnings.filterwarnings('ignore')

# Stopwords list
stop_words = set(stopwords.words('english'))

# Stemmer
stemmer = PorterStemmer()

# Contractions mapping
contractions_mapping = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "when's": "when is",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

# Emoji mapping
emoji_mapping = {
    ":)" : "happy",
    ":(" : "sad",
    ";)" : "wink",
    ":/" : "meh"
}

def replace_words(string, dict):
    pattern = r'(?<!\w)(?:' + '|'.join(re.escape(word) for word in dict.keys()) + r')(?!\w)'
    def replace(match):
        return dict.get(match.group(0), match.group(0))
    return re.sub(pattern, replace, string)

# Preprocess function incorporating additional steps
def preprocess(text, use_stemming=True):
    # Lowercase
    text = text.lower()
    # Replace URLs with a placeholder
    text = re.sub(r'http\S+', 'http', text)
    # Replace user mentions with a placeholder
    text = re.sub(r'@\w+', '@user', text)
    # Expand contractions
    text = replace_words(text, contractions_mapping)
    # Replace ASCII emojis
    text = replace_words(text, emoji_mapping)
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Expand contractions again without special characters
    c_mapping_no_specials = {}
    for key, value in contractions_mapping.items():
        new_key = re.sub(r'[^\w\s]', '', key)
        c_mapping_no_specials[new_key] = value
    c_mapping_no_specials.pop("hell")
    text = replace_words(text, c_mapping_no_specials)
    # Tokenize
    tokens = word_tokenize(text)
    # Optionally apply stemming
    if use_stemming:
        filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    else:
        filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join back into a string, preserving emoticons
    return " ".join(filtered_tokens)

# Test the updated function with a sample tweet, both with and without stemming
# sample_tweet = "@user I can't believe it's already summer! Time flies so fast 😱 http://example.com"
# preprocessed_tweet_with_stemming = preprocess(sample_tweet, use_stemming=True)
# preprocessed_tweet_without_stemming = preprocess(sample_tweet, use_stemming=False)
# print(preprocessed_tweet_with_stemming)
# print(preprocessed_tweet_without_stemming)

# Load and preprocess the data
train_texts = open('../data/tweeteval/sentiment/train_text.txt').read().strip().split('\n')
train_labels = open('../data/tweeteval/sentiment/train_labels.txt').read().strip().split('\n')
val_texts = open('../data/tweeteval/sentiment/val_text.txt').read().strip().split('\n')
val_labels = open('../data/tweeteval/sentiment/val_labels.txt').read().strip().split('\n')
test_texts = open('../data/tweeteval/sentiment/test_text.txt').read().strip().split('\n')
test_labels = open('../data/tweeteval/sentiment/test_labels.txt').read().strip().split('\n')
mapping_lines = open('../data/tweeteval/sentiment/mapping.txt').read().strip().split('\n')
mapping_dict = {line.split()[0]: " ".join(line.split()[1:]) for line in mapping_lines}
mapped_train_labels = [mapping_dict[label.strip()] for label in train_labels]
mapped_val_labels = [mapping_dict[label.strip()] for label in val_labels]
mapped_test_labels = [mapping_dict[label.strip()] for label in test_labels]

# Apply preprocessing
preprocessed_train_texts_data = [preprocess(text) for text in train_texts]
preprocessed_val_texts_data = [preprocess(text) for text in val_texts]
preprocessed_test_texts_data = [preprocess(text) for text in test_texts]

print("Data preprocessing complete!")

# Create DataFrame with simplified preprocessing
df_train = pd.DataFrame({
    'Text': preprocessed_train_texts_data,
    'Label_ID': train_labels,
    'Mapped_Label': mapped_train_labels
})
df_val = pd.DataFrame({
    'Text': preprocessed_val_texts_data,
    'Label_ID': val_labels,
    'Mapped_Label': mapped_val_labels
})
df_test = pd.DataFrame({
    'Text': preprocessed_test_texts_data,
    'Label_ID': test_labels,
    'Mapped_Label': mapped_test_labels
})

print("################ Train Data Preview: #####################")
print(df_train.head()) # Check the first few rows of the DataFrame
print("##########################################################")

print("################ Validation Data Preview: ################")
print(df_val.head()) # Check the first few rows of the DataFrame
print("##########################################################")

print("################ Test Data Preview: ######################")
print(df_test.head()) # Check the first few rows of the DataFrame
print("##########################################################")

# Make CSV File for preprocessed data prior to BERT tokenization
os.makedirs('../data/tweeteval/sentiment/csv', exist_ok=True)  
df_train.to_csv('../data/tweeteval/sentiment/csv/pre-token-train-data.csv', index=False)
df_val.to_csv('../data/tweeteval/sentiment/csv/pre-token-val-data.csv', index=False)
df_test.to_csv('../data/tweeteval/sentiment/csv/pre-token-test-data.csv', index=False)

print("Preprocessed data saved to CSV files!")
