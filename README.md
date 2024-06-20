# Question-Classification-and-Named-Entity-Recognition-NER-Project
## Description

This project consists of two main components: question classification and named entity recognition (NER) using machine learning techniques.

## Question Classification:

The dataset (question_classif.csv) contains a series of questions related to various courses and their corresponding labels. The primary goal is to classify these questions into different categories such as question_rag.

## Named Entity Recognition (NER):

The dataset (test.csv) includes sentences with tokenized words and corresponding labels. The objective is to identify and classify entities within the sentences, such as names of persons or specific tasks.

# Files
main.ipynb: A Jupyter Notebook that presumably contains the main implementation of the project, including data loading, preprocessing, model training, and evaluation for both question classification and NER.

train_ner_model.ipynb: A Jupyter Notebook dedicated to training the NER model, including steps for data preparation, model architecture, training process, and evaluation metrics.

question_classif.csv: The dataset for question classification containing columns for questions, their text labels, and numerical labels.

test.csv: The dataset for NER tasks containing columns for sentences, tokenized words, corresponding labels, and task types.

# Usage

## Question Classification:

Load the question_classif.csv dataset.
Preprocess the data to prepare it for model training.
Train a classification model to categorize questions into predefined labels.
Evaluate the model's performance using appropriate metrics.

## NER:

* Load the test.csv dataset.
* Preprocess the data, ensuring proper tokenization and labeling.
* Train an NER model to identify and classify entities within the sentences.
* Evaluate the model's performance using appropriate NER metrics such as precision, recall, and F1-score.

# Requirements

* Python
* Jupyter Notebook
* pandas
* sklearn (for classification)
* spaCy or similar library (for NER)

# Getting Started

Clone the repository and navigate to the project directory.
Open the Jupyter Notebooks (main.ipynb and train_ner_model.ipynb) to explore and run the code.
Follow the steps outlined in the notebooks to preprocess the data, train the models, and evaluate their performance.