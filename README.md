# Bankruptcy Prediction on Imbalanced Financial Data

This repository contains a comprehensive machine learning pipeline designed to predict company bankruptcy using financial indicators. The project explores the critical challenges of real-world financial datasets: extreme class imbalance, high-dimensionality, and the high asymmetric cost of Type 2 errors (false negatives).

## က Project Overview

Lenders and investors face asymmetric financial consequences: failing to flag a company heading towards bankruptcy (a Type 2 error) is significantly more expensive than misclassifying a healthy company as risky (a Type 1 error). 

This project evaluates and compares **six machine learning models** across **three balancing configurations** (Original, SMOTE oversampling, and Random Undersampling) to determine the best approach for minimizing costly false negatives while maintaining reasonable precision.

### Key Highlights
* **Extreme Class Imbalance:** Handles a dataset where only ~3.2% (220 out of 6,819) of the companies are bankrupt.
* **Dimensionality Reduction:** Prunes redundant, highly-correlated indicators (correlation > 0.95) and uses **Random Forest Feature Importance** to select the top 30 key predictors from an original 94 indicators, maintaining **77.3% of total feature importance** while significantly reducing training times.
* **Feature Transformation:** Addresses heavily skewed features containing extreme outliers (ratios with ranges up to 10 billion) using a non-parametric `QuantileTransformer`.
* **Robustness Check:** Validates results over multiple random data-splitting seeds to ensure stability of the final recommendations.

---

## ကက Tech Stack & Libraries
* **Languages:** Python 3
* **Core Libraries:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`
* **Imbalanced Learning:** `imbalanced-learn` (`SMOTE`, `RandomUnderSampler`)

---

## က Models Evaluated
1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Tree**
4. **MLP Neural Network**
5. **Random Forest**
6. **Gradient Boosting**

---

## က Key Findings & Insights

* **The Accuracy Paradox:** Using accuracy alone as a metric is highly misleading. On the original unbalanced dataset, always predicting "not bankrupt" achieves **96.77% accuracy** but fails to catch a single bankrupt company.
* **The Balance Trade-Off:** 
  * **Undersampling** yields the highest overall recall (~84-88% bankruptcies caught) but creates a massive volume of false alarms (Type 1 errors).
  * **SMOTE oversampling** achieves a much healthier trade-off between recall (sensitivity) and precision.
* **Best Overall Performer:** The **Random Forest model combined with SMOTE** achieved the best overall balance (**F1-Score of 0.514**, catching 27/44 bankruptcies with only 34 false alarms on the test set). This trend holds stable during multi-seed robustness testing.

---

## က How to Run the Notebook

1. Clone this repository to your local system or open the notebook directly in **Google Colab**.
2. Upload the `Dataset.csv` containing the company financial records.
3. Verify the `data_path` parameter matches your file storage directory in the parameters cell.
4. Run the notebook from top to bottom.
