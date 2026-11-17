# public-health-AI-risk-prediction
Reproducible ML pipeline in Python using Random Forest (scikit-learn) to model public health risk from epidemiological and environmental data. Features ROC-AUC evaluation and feature importance analysis for chronic disease surveillance and transparent risk stratification. What is SICKIT learn here

# AI & Machine Learning in Public Health: Predictive Risk Modeling

## Project Overview
This repository demonstrates a machine learning pipeline designed for public health surveillance. It applies a classification model to evaluate patient or cohort risk factors for chronic disease outcomes using synthetic epidemiological data.

## Key Methodology & Workflow
* **Data Preprocessing & Imputation:** Handling missing epidemiological features and normalizing continuous variables.
* **Predictive Modeling:** Training a Random Forest Classifier to estimate individual risk probability based on lifestyle, environmental, and physiological indicators.
* **Model Evaluation:** Utilizing ROC-AUC, precision-recall metrics, and feature importance plotting to ensure clinical interpretability.

## Repository Contents
* `public_health_risk_model.py`: Python script containing data generation, model training, and performance evaluation.

## Tools & Libraries
* **Language:** Python
* **Packages:** `scikit-learn`, `pandas`, `numpy`
