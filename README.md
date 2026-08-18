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

**GitHub Repository:** https://github.com/priyathara1801-create/bank-marketing-ml-assignment

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
| ML Model Name                        | Observation about model performance                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**              | Shows strong overall performance, with the highest accuracy among the five models and a very strong AUC of 0.9424. Its precision is also good, meaning that many of the customers predicted to subscribe actually do subscribe. However, its recall of 0.4364 shows that it still misses a noticeable number of actual positive subscriptions. Overall, it provides a strong and reliable balance, especially in terms of accuracy and discrimination ability.                                                                |
| **Decision Tree**                    | Gives the best F1 Score of all five models, which suggests that it provides the strongest balance between precision and recall on this dataset. Its recall is also relatively high compared with Logistic Regression, kNN, and Random Forest, allowing it to identify more customers who actually subscribe. Although its accuracy and AUC are slightly lower than Logistic Regression and Random Forest, its balanced performance makes it a strong choice when both false positives and missed subscriptions are important. |
| **kNN**                              | Performs reasonably well, with an accuracy of 0.9082 and an AUC of 0.8993. However, its recall and F1 Score are lower than those of Logistic Regression and Decision Tree, meaning that it misses a larger number of actual subscribers. While kNN provides acceptable overall performance, it does not achieve the same balance between identifying positive cases and maintaining precision as the better-performing models.                                                                                                |
| **Naive Bayes**                      | An interesting case because it achieves the highest recall of all five models at 0.6907. This means it is the best model for identifying customers who actually subscribe to the term deposit. However, this comes at the cost of much lower precision, accuracy, and F1 Score, indicating that it also produces many false positives. In other words, it is more aggressive at predicting subscriptions, but less reliable overall when compared with the other models.                                                      |
| **Random Forest (Ensemble)**         | Achieves the highest precision of 0.7809 and also has the strongest AUC of 0.9435, slightly outperforming Logistic Regression in terms of discrimination ability. However, its recall is the lowest at 0.2996, meaning that it misses a large number of actual subscribers. So, while Random Forest is highly reliable when it predicts a positive subscription, it is too conservative in identifying all possible positive cases.                                                                                           |
| **Overall Winner for your dataset?** | **Decision Tree.** It achieves the highest F1 Score (0.5742), providing the best balance between Precision and Recall among the five models. Since the Bank Marketing dataset has an imbalanced target variable, accuracy alone does not give the complete picture. Decision Tree identifies positive subscriptions more effectively while maintaining a reasonable precision level, making it the best overall model based on the balance of the evaluation metrics.                                                         |


## E. Streamlit Application

The Streamlit application provides the following required features:

- CSV test-data upload
- Model-selection dropdown
- Evaluation metrics
- Confusion matrix
- Classification report
- Prediction preview

### Live Streamlit App Link

**Streamlit App:** https://bank-marketing-ml-assignment-93mk17p9x24lr7ym7wgfzt.streamlit.app/

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
