# Import necessary libraries for training, validating and testing the baseline model
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import time
import os
import re

# Load the data
df_train = pd.read_csv('../data/tweeteval/sentiment/csv/pre-token-train-data.csv')
df_val = pd.read_csv('../data/tweeteval/sentiment/csv/pre-token-val-data.csv')
df_test = pd.read_csv('../data/tweeteval/sentiment/csv/pre-token-test-data.csv')

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

# Save post-tokenized data separately [NOT SAVING FOR NOW SINCE THEY ARE TOO BIG]
#torch.save(train_data, '../data/tweeteval/sentiment/csv/post-token-train-data.pt')
#torch.save(val_data, '../data/tweeteval/sentiment/csv/post-token-val-data.pt')
#torch.save(test_data, '../data/tweeteval/sentiment/csv/post-token-test-data.pt')

print("Data tokenization completed!")

class TweetEvalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": item[0],
            "attention_mask": item[1],
            "targets": item[2]
        }

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)
    
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    
    return correct_predictions.double() / n_examples, np.mean(losses)

train_dataset = TweetEvalDataset(train_data)
val_dataset = TweetEvalDataset(val_data)
test_dataset = TweetEvalDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentimentClassifier(n_classes=3)  # Adjust the number of classes as per your dataset
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 10
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    print('-' * 10)
    
    train_acc, train_loss = train_epoch(
        model,
        train_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(train_data)
    )
    
    print(f'Train loss {train_loss} accuracy {train_acc}')
    
    val_acc, val_loss = eval_model(
        model,
        val_loader,
        loss_fn,
        device,
        len(val_data)
    )
    
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

# Evaluation on test data
test_acc, _ = eval_model(
    model,
    test_loader,
    loss_fn,
    device,
    len(test_data)
)

print(f'Test accuracy: {test_acc.item()}')
