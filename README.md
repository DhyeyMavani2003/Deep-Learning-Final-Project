# DeepSentiment: Tweet Sentiment Analysis using BERT-based models and LLMs

## Authors
- Dhyey Mavani, Amherst College'25
- Carl May, Williams College'25

## Professor & Project Advisor
- Professor Bálint Gyires-Tóth, AIT Budapest and NVIDIA

## Abstract

This project aims to analyze sentiment in tweets data using Deep Learning models, specifically BERT. The process involves cleaning and preprocessing the tweets data, followed by fine-tuning the BERT model to improve sentiment prediction accuracy. Additionally, other Deep Learning models will be explored to compare their performance with BERT. By leveraging the power of Deep Learning, the project seeks to gain insights into the sentiment patterns present in the tweets data and provide valuable information for sentiment analysis tasks.

## Tools and Technologies Used
 Please refer to `requirements.txt` for requirements related to running `.py` files in the project
- Python
- Numpy
- Pandas
- Tensorflow
- OS and System Libraries
- GitHub Copilot

## Instructions to Run The Project

1. **Data Glancing:** You can start with inspecting the data that we used on the `./data/tweeteval/sentiment` relative path. You can see a `csv` folder there, which has files that are generated through our preprocessing phase described further below.

2. **Preprocessing:** You can start by navigating to `./code` relative path. There you will see our preprocessing script `preprocess_tweeteval.py`. In order to execute this Python script you can run the command `python preprocess_tweeteval.py` on your terminal. This will run the script, which in turn accesses the raw `.txt` data, preprocesses it, and initializes the appropriate csv files at `./data/tweeteval/sentiment/csv`

3. **Tokenization and Model(s) Training, Validation, Testing:** For checkpointed and saved code results as a snapshot, please feel free to navigate to `./code/New_tokenization_and_naive_bayes_and_bert_model_train_val_test.ipynb`, there you will also have the ability to open the same with Google Colab and reproduce our results.

## License

This work is available under the Apache license as detailed in the `./` root directory of the project through the LICENSE file.
