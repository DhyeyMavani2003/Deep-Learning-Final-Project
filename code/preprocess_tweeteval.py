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

def expand_contractions(text, contractions_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contractions_mapping.get(match) if contractions_mapping.get(match) else contractions_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    
    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text

def replace_emoji(string, emoji_dict):
    pattern = r'(?<!\w)(?:' + '|'.join(re.escape(emoji) for emoji in emoji_dict.keys()) + r')(?!\w)'
    def replace(match):
        return emoji_dict.get(match.group(0), match.group(0))
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
    text = expand_contractions(text, contractions_mapping)
    # Replace ASCII emojis
    text = replace_emoji(text, emoji_mapping)
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Expand contractions again without special characters
    c_mapping_no_specials = {}
    for key, value in contractions_mapping.items():
        new_key = re.sub(r'[^\w\s]', '', key)
        c_mapping_no_specials[new_key] = value
    c_mapping_no_specials.pop("hell")
    text = expand_contractions(text, c_mapping_no_specials)
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

# Load the BERT tokenizer.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize all of the sentences and map the tokens to their word IDs for df_train
train_input_ids = []
train_attention_masks = []

for sent in df_train['Text']:
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=256,
                        pad_to_max_length=True,
                        return_attention_mask=True,
                        return_tensors='pt',
                   )
    
    train_input_ids.append(encoded_dict['input_ids'])
    train_attention_masks.append(encoded_dict['attention_mask'])

train_input_ids = torch.cat(train_input_ids, dim=0)
train_attention_masks = torch.cat(train_attention_masks, dim=0)
train_labels = torch.tensor(df_train['Label_ID'].values.astype(int))

train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

# Preview the tokenized train data
print("################ Tokenized Train Data Preview: #####################")
print(train_data[0]) # Check the first few rows of the DataFrame
print("###################################################################")

# Tokenize all of the sentences and map the tokens to their word IDs for df_val
val_input_ids = []
val_attention_masks = []

for sent in df_val['Text']:
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=256,
                        pad_to_max_length=True,
                        return_attention_mask=True,
                        return_tensors='pt',
                   )
    
    val_input_ids.append(encoded_dict['input_ids'])
    val_attention_masks.append(encoded_dict['attention_mask'])

val_input_ids = torch.cat(val_input_ids, dim=0)
val_attention_masks = torch.cat(val_attention_masks, dim=0)
val_labels = torch.tensor(df_val['Label_ID'].values.astype(int))

val_data = TensorDataset(val_input_ids, val_attention_masks, val_labels)
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=True)

# Preview the tokenized validation data
print("################ Tokenized Validation Data Preview: ################")
print(val_data[0]) # Check the first few rows of the DataFrame
print("###################################################################")

# Tokenize all of the sentences and map the tokens to their word IDs for df_test
test_input_ids = []
test_attention_masks = []

for sent in df_test['Text']:
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=256,
                        pad_to_max_length=True,
                        return_attention_mask=True,
                        return_tensors='pt',
                   )
    
    test_input_ids.append(encoded_dict['input_ids'])
    test_attention_masks.append(encoded_dict['attention_mask'])

test_input_ids = torch.cat(test_input_ids, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)
test_labels = torch.tensor(df_test['Label_ID'].values.astype(int))

test_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

# Preview the tokenized test data
print("################ Tokenized Test Data Preview: ######################")
print(test_data[0]) # Check the first few rows of the DataFrame
print("###################################################################")

# Save post-tokenized data separately 
torch.save(train_data, '../data/tweeteval/sentiment/csv/post-token-train-data.pt')
torch.save(val_data, '../data/tweeteval/sentiment/csv/post-token-val-data.pt')
torch.save(test_data, '../data/tweeteval/sentiment/csv/post-token-test-data.pt')

print("Data tokenization and saving completed.")
