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
