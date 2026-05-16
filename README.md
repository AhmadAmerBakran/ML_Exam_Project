# Adult Income Classification - Machine Learning Mini Project

**Course:** PBSW - Machine Learning - Spring 2026  
**Project type:** Fundamental Machine Learning model architectures  
**Group members:** Ahmad Amer Bakran and Mahmoud Eybo

## Project purpose

This project trains, fine-tunes, evaluates, and compares two machine learning model architectures on the Adult Census Income dataset.

The task is a binary classification problem:

> Predict whether a person earns more than 50K USD per year (`>50K`) or not (`<=50K`) based on census attributes such as age, work class, education level, occupation, marital status, capital gain/loss, and weekly working hours.

## Dataset

Dataset: **Adult / Census Income** from the UCI Machine Learning Repository.

The notebook downloads the data directly from UCI:

- `adult.data` is used as the training data.
- `adult.test` is used as the final test data.

## Model architectures

The project examines and compares two model architectures from the allowed list:

1. **Random Forest Classifier**
2. **Histogram-Based Gradient Boosting Classifier**

Both models are trained inside scikit-learn pipelines and fine-tuned using `RandomizedSearchCV`.

## Data preparation

The data preparation is intentionally non-trivial because the dataset is tabular. The notebook includes:

- Loading the official train and test files.
- Cleaning target labels.
- Replacing missing category values marked as `?`.
- Removing duplicate rows from the training data.
- Feature engineering:
  - `capital_net`
  - `has_capital_gain`
  - `has_capital_loss`
- Dropping selected columns:
  - `education`, because `education_num` already contains the ordered education level.
  - `fnlwgt`, because it is a census sampling weight and less suitable as a normal predictive feature.
- Numeric preprocessing:
  - median imputation
- Categorical preprocessing:
  - most-frequent imputation
  - one-hot encoding

## Evaluation

The models are compared using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report
- ROC curve

The notebook saves the best model and comparison results after it has been run.
