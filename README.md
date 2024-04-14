# DeepSentiment: Tweet Sentiment Analysis using BERT and Deep Learning

## Authors
- Dhyey Mavani
- Carl May

## Abstract

This project aims to analyze sentiment in tweets data using Deep Learning models, specifically BERT. The process involves cleaning and preprocessing the tweets data, followed by fine-tuning the BERT model to improve sentiment prediction accuracy. Additionally, other Deep Learning models will be explored to compare their performance with BERT. By leveraging the power of Deep Learning, the project seeks to gain insights into the sentiment patterns present in the tweets data and provide valuable information for sentiment analysis tasks.

## Tools and Technologies Used
- Python
- Numpy
- Pandas
- Tensorflow
- OS and System Libraries
- GitHub Copilot

## Instructions to Run The Project

1. **Data Glancing:** You can start with inspecting the data that we used on the `./data/tweeteval/sentiment` relative path. You can see a `csv` folder there, which has files that are generated through our preprocessing phase described further below.

2. **Preprocessing:** You can start by navigating to `./code` relative path. There you will see our preprocessing script `preprocess_tweeteval.py`. In order to execute this Python script you can run the command `python preprocess_tweeteval.py` on your terminal. This will run the script, which in turn accesses the raw `.txt` data, preprocesses it, and initializes the appropriate csv files at `./data/tweeteval/sentiment/csv`

3. **Tokenization and Baseline Model Training, Validation, Testing:** You can refer to the Google Colab notebook, which has the entire code for the same at https://colab.research.google.com/drive/1p4YI8XAXpCA_s4C5RadpVkIt9hEqY3N8?usp=sharing

