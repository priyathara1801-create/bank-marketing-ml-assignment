# Bank Marketing Classification — Machine Learning Assignment 2

## A. Problem Statement

The objective of this project is to implement classification models for predicting whether a customer will subscribe to a term deposit (`yes` or `no`) using the Bank Marketing dataset.

Five classification models are trained and evaluated on the same dataset. The evaluation metrics are Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

An interactive Streamlit application is also provided for evaluating the models on test data.

## B. Dataset Description

The dataset used is the **Bank Marketing dataset**, supplied as `bank-additional-full.csv`.

The dataset contains:

- 41,188 instances
- 20 input features
- 1 binary target variable (`y`)
- Target classes: `yes` and `no`

The input features contain customer, campaign/contact and economic information.

The target variable `y` was converted to binary form:

- `yes` = 1
- `no` = 0

An 80:20 stratified train/test split was used with `random_state=42`.

The `test_data.csv` file contains the held-out test data used by the Streamlit application.

## C. GitHub Repository Link

**GitHub Repository:** `[PASTE YOUR GITHUB REPOSITORY LINK HERE]`

## D. Models Used

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest Ensemble

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.9087 | 0.8535 | 0.6050 | 0.5463 | 0.5742 | 0.5241 |
| Random Forest | 0.9116 | 0.9435 | 0.7809 | 0.2996 | 0.4330 | 0.4492 |
| Logistic Regression | 0.9166 | 0.9424 | 0.7118 | 0.4364 | 0.5411 | 0.5162 |
| kNN | 0.9082 | 0.8993 | 0.6419 | 0.4192 | 0.5072 | 0.4717 |
| Naive Bayes | 0.8203 | 0.8393 | 0.3495 | 0.6907 | 0.4642 | 0.4009 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Decision Tree | Accuracy=0.9087, AUC=0.8535, Precision=0.6050, Recall=0.5463, F1=0.5742, MCC=0.5241. |
| Random Forest | Accuracy=0.9193, AUC=0.9466, Precision=0.7097, Recall=0.4795, F1=0.5723, MCC=0.5420. |
| Logistic Regression | Accuracy=0.9166, AUC=0.9424, Precision=0.7118, Recall=0.4364, F1=0.5411, MCC=0.5162. |
| kNN | Accuracy=0.9082, AUC=0.8993, Precision=0.6419, Recall=0.4192, F1=0.5072, MCC=0.4717. |
| Naive Bayes | Accuracy=0.8203, AUC=0.8393, Precision=0.3495, Recall=0.6907, F1=0.4642, MCC=0.4009. |

### Overall Winner

**Decision Tree** achieved the highest F1 score on the held-out test data used in this project.

The final winner should be interpreted using the complete set of evaluation metrics rather than Accuracy alone because the target classes are imbalanced.

## E. Streamlit Application

The Streamlit application provides the following required features:

- CSV test-data upload
- Model-selection dropdown
- Evaluation metrics
- Confusion matrix
- Classification report
- Prediction preview

### Live Streamlit App Link

**Streamlit App:** `[PASTE YOUR STREAMLIT APP LINK HERE]`

## Project Structure

```text
bank-marketing-assignment/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_metrics.csv
│
└── model/
    ├── train_models.py
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

## How to Run the Application

Install the required packages:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

Upload `test_data.csv` and select a model from the sidebar.

## Reproducibility

- Train/test split: 80/20
- Stratification: Yes
- Random state: 42
- Numerical preprocessing: median imputation + standardization
- Categorical preprocessing: most-frequent imputation + one-hot encoding
- Logistic Regression: max_iter=1000
- Decision Tree: max_depth=12
- kNN: 7 neighbors
- Random Forest: 40 trees, max_depth=10
