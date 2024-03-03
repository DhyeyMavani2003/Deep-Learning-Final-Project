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
    "i'd": "I would",
    "i'll": "I will",
    "i'm": "I am",
    "i've": "I have",
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
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
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
mapping_lines = open('../data/tweeteval/sentiment/mapping.txt').read().strip().split('\n')
mapping_dict = {line.split()[0]: " ".join(line.split()[1:]) for line in mapping_lines}
mapped_labels = [mapping_dict[label.strip()] for label in train_labels]

# Apply preprocessing
preprocessed_data = [preprocess(text) for text in train_texts]

# Create DataFrame with simplified preprocessing
df = pd.DataFrame({
    'Text': preprocessed_data,
    'Label_ID': train_labels,
    'Mapped_Label': mapped_labels
})

print(df.head()) # Check the first few rows of the DataFrame

# Load the BERT tokenizer.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize all of the sentences and map the tokens to their word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in df['Text']:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(df['Label_ID'].values.astype(int))

# Create the DataLoader for the dataset.
data = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(data, batch_size=32, shuffle=True)
print(data.tensors)