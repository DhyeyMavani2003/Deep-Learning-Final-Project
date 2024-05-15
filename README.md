# DeepSentiment: Tweet Sentiment Analysis using BERT-based models & LLMs

## Authors
- Dhyey Mavani, Amherst College'25
- Carl May, Williams College'25

## Professor & Project Advisor
- Professor Bálint Gyires-Tóth, AIT Budapest and NVIDIA

## Abstract

This project, DeepSentiment, leverages advanced deep learning techniques, particularly BERT (Bidirectional Encoder Representations from Transformers) and LLMs (Large Language Models), to analyze and predict sentiment from tweets. The primary aim is to perform sentiment analysis using BERT-based models, and observe the effectiveness of other contemporary LLM-based techniques such as multi-shot learning in classification of the entries. The insights garnered from tweet sentiment patterns can enhance understanding and implementation of state-of-the-art deep learning techniques in diverse sentiment analysis applications such as financial markets, and behavioral economics.

## Tools and Technologies Used
- Python
- Numpy
- Pandas
- Transformers
- Tensorflow
- NLTK
- OS and System Libraries
- GitHub Copilot

_Note:_  Please refer to `requirements.txt` for requirements related to running `.py` files in the project.

## Instructions to Run The Project

1. **Data Glancing:** You can start with inspecting the data that we used on the `./data/tweeteval/sentiment` relative path. You can see a `csv` folder there, which has files that are generated through our preprocessing phase described further below.

2. **Preprocessing:** You can start by navigating to `./code` relative path. There you will see our preprocessing script `preprocess_tweeteval.py`. In order to execute this Python script you can run the command `python preprocess_tweeteval.py` on your terminal. This will run the script, which in turn accesses the raw `.txt` data, preprocesses it, and initializes the appropriate csv files at `./data/tweeteval/sentiment/csv`

3. **Tokenization and Model(s) Training, Validation, Testing:** For checkpointed and saved code results as a snapshot, please feel free to navigate to `./code/tokenization_naive_bayes_and_bert_models_train_val_test_code.ipynb`, there you will also have the ability to open the same with Google Colab and reproduce our results.
  
4. **Using LLM Model(s) through Multi-shot Learning:** For checkpointed and saved code results as a snapshot, please feel free to navigate to `./code/Phi_Sentiment.ipynb`, there you will also have the ability to open the same with Google Colab and reproduce our results.

## License

This work is available under the Apache license as detailed in the `./` root directory of the project through the LICENSE file.
