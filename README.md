# Fraud Detection System — PaySim Mobile Money Transactions

A machine learning pipeline that detects fraudulent mobile-money transactions on the PaySim dataset (~6.36M transactions), comparing multiple classification algorithms and evaluating the impact of SMOTE on a heavily imbalanced target (~0.13% fraud).

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Approach](#approach)
  - [1. Exploratory Data Analysis](#1-exploratory-data-analysis)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Handling Class Imbalance](#3-handling-class-imbalance)
  - [4. Model Comparison](#4-model-comparison)
  - [5. Final Model](#5-final-model)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [License](#license)

## Overview

Fraud detection is a classic rare-event classification problem: fraudulent transactions make up a tiny fraction of all transactions, so a naive model can score 99%+ accuracy while catching almost no fraud. This project works through the full pipeline for that kind of problem:

- Exploratory data analysis to understand transaction types, amounts, and fraud patterns
- Feature engineering based on account balance inconsistencies
- A controlled comparison of several algorithms on Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC
- An evaluation of SMOTE (Synthetic Minority Over-sampling Technique) for handling the class imbalance
- A final production model trained on the full dataset

## Dataset

**Source:** [PaySim — Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)

PaySim simulates mobile money transactions based on real transaction logs from an African mobile money service, injected with fraudulent behavior.

| Property | Value |
|---|---|
| Rows | ~6.36 million |
| Columns | 11 |
| Target | `isFraud` (binary) |
| Fraud rate | ~0.13% |
| Transaction types | `PAYMENT`, `TRANSFER`, `CASH_OUT`, `CASH_IN`, `DEBIT` |

| Column | Description |
|---|---|
| `step` | Simulated time unit (1 step = 1 hour) |
| `type` | Transaction type |
| `amount` | Transaction amount |
| `nameOrig` | Customer initiating the transaction |
| `oldbalanceOrg` / `newbalanceOrig` | Sender's balance before / after the transaction |
| `nameDest` | Recipient of the transaction |
| `oldbalanceDest` / `newbalanceDest` | Recipient's balance before / after the transaction |
| `isFraud` | Ground-truth fraud label (target) |
| `isFlaggedFraud` | A rule-based flag from the original simulation (not used as a feature) |

> The raw `Fraud.csv` file is not included in this repo due to its size — download it from the Kaggle link above and place it in the project root before running the notebook.

## Project Structure

```
├── fraud_detection.ipynb     # main notebook: EDA, feature engineering, model comparison, SMOTE, final model
├── models/
│   └── fraud_hgb_model.pkl   # saved production model (generated after running the notebook)
├── requirements.txt
├── README.md
└── Fraud.csv                 # not included — download separately (see Dataset section)
```

## Approach

### 1. Exploratory Data Analysis

- Class distribution of `isFraud` (count + proportion)
- Transaction volume and fraud rate by `type` — fraud is concentrated almost entirely in `TRANSFER` and `CASH_OUT` transactions
- Transaction amount distribution (log scale) and amount comparison between fraud and legitimate transactions
- Fraud frequency over simulated time (`step`)
- Correlation analysis on the numerical features

### 2. Feature Engineering

- `errorBalanceOrig` = `newbalanceOrig + amount - oldbalanceOrg`
- `errorBalanceDest` = `oldbalanceDest + amount - newbalanceDest`

These capture inconsistencies between what an account balance *should* be after a transaction and what it actually is — a strong indicator of fraudulent activity in this dataset.

- Dropped `nameOrig` / `nameDest` (identifiers, not predictive on their own) and `isFlaggedFraud` (a static rule from the original simulation)
- One-hot encoded `type`

### 3. Handling Class Imbalance

Two strategies were evaluated:

- **No resampling** — training directly on the natural (imbalanced) class distribution
- **SMOTE** — applied only to the training split (never to the test set, to avoid data leakage), generating synthetic fraud examples by interpolating between existing fraud cases and their nearest neighbors

Because retraining several algorithms with SMOTE-inflated data on the full 6.36M-row dataset isn't practical, the algorithm comparison and SMOTE experiment run on a large stratified sample (every fraud transaction + a random sample of legitimate ones), while the final production model is trained on the full dataset.

### 4. Model Comparison

The following algorithms were trained and evaluated, with and without SMOTE:

- Logistic Regression
- Random Forest
- XGBoost (falls back to Gradient Boosting if XGBoost isn't installed)
- HistGradientBoosting

Each was scored on Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC.

### 5. Final Model

`HistGradientBoostingClassifier`, trained on the full dataset, chosen for its speed on large tabular data and its performance in the comparison above. `class_weight='balanced'` was deliberately **not** used in the final model, to avoid over-penalizing precision — a false positive here means blocking a real customer's legitimate transaction.
- Accuracy is not a meaningful metric under ~0.13% fraud — a model predicting "no fraud" for everything already scores ~99.87%. PR-AUC is the primary metric used for model selection.
- Fraud in this dataset only occurs in `TRANSFER` and `CASH_OUT` transactions.
- `errorBalanceOrig` / `errorBalanceDest` are strong, engineered fraud signals.
 
## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

Download `Fraud.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/ealaxi/paysim1) and place it in the project root.

## Usage

```bash
jupyter notebook fraud_detection.ipynb
```

Run all cells in order. The final trained model is saved to `models/fraud_hgb_model.pkl`.

## Tech Stack

- Python, pandas, NumPy
- scikit-learn
- imbalanced-learn (SMOTE)
- XGBoost
- Matplotlib, Seaborn
- Jupyter Notebook

 
## License

This project is for educational purposes. Dataset licensed by its original authors on Kaggle. 
