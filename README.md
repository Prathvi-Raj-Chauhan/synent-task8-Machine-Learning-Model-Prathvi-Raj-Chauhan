# House Price Prediction using Machine Learning

## Problem Statement

The objective of this project is to build a machine learning model capable of predicting house prices based on various property features such as area, number of bedrooms, bathrooms, parking availability, furnishing status, and other amenities.

Accurate house price prediction can assist buyers, sellers, and real estate businesses in making informed decisions based on historical housing data.

---

## Dataset Details

**Dataset:** Housing Price Prediction Dataset

**Source:** Kaggle

**Link:** https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction

### Dataset Overview

* Total Records: **545**
* Total Features: **13**
* Target Variable: **price**

### Features

| Feature          | Description                         |
| ---------------- | ----------------------------------- |
| price            | House price (Target Variable)       |
| area             | Total area of the house             |
| bedrooms         | Number of bedrooms                  |
| bathrooms        | Number of bathrooms                 |
| stories          | Number of floors/stories            |
| mainroad         | Whether connected to the main road  |
| guestroom        | Availability of guest room          |
| basement         | Availability of basement            |
| hotwaterheating  | Availability of hot water heating   |
| airconditioning  | Availability of air conditioning    |
| parking          | Number of parking spaces            |
| prefarea         | Whether located in a preferred area |
| furnishingstatus | Furnishing status of the house      |

---

## Approach

### 1. Data Preprocessing

* Loaded the dataset using KaggleHub.
* Standardized column names by converting them to lowercase and replacing spaces with underscores.
* Converted binary categorical features (`yes/no`) into numerical values (`1/0`).
* Separated the dataset into:

  * Features (`X`)
  * Target (`y`)

### 2. Feature Identification

Features were categorized into:

* **Numerical Features**

  * area
  * bedrooms
  * bathrooms
  * stories
  * parking

* **Categorical Features**

  * mainroad
  * guestroom
  * basement
  * hotwaterheating
  * airconditioning
  * prefarea
  * furnishingstatus

### 3. Data Preprocessing Pipeline

#### Numerical Features

* Missing value imputation using median values
* Feature scaling using StandardScaler

#### Categorical Features

* Missing value imputation using the most frequent value
* One-Hot Encoding for categorical variables

### 4. Train-Test Split

The dataset was divided into:

* Training Set: 80%
* Testing Set: 20%

```python
train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

### 5. Model Training

Two machine learning models were trained and evaluated:

#### Linear Regression

* Baseline regression model
* Feature selection performed using SelectKBest with F-test scores

#### Random Forest Regressor

* Ensemble-based regression model
* 200 decision trees
* Random state fixed for reproducibility

### 6. Evaluation Metrics

The following metrics were used:

#### RMSE (Root Mean Squared Error)

Measures the average prediction error in the same units as house prices.

[
RMSE = \sqrt{\frac{\sum(y - \hat{y})^2}{n}}
]

#### R² Score (Coefficient of Determination)

Measures how well the model explains the variance in house prices.

[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
]

---

## Results

### Linear Regression

| Metric   | Value        |
| -------- | ------------ |
| RMSE     | 1,324,506.96 |
| R² Score | 0.6529       |

### Random Forest Regressor

| Metric   | Value        |
| -------- | ------------ |
| RMSE     | 1,393,929.51 |
| R² Score | 0.6156       |

---

## Analysis

* The Linear Regression model achieved the best performance on the test dataset.
* It obtained a lower RMSE and higher R² score compared to the Random Forest model.
* Approximately **65.29%** of the variance in house prices was explained by the Linear Regression model.
* The Random Forest model performed reasonably well but did not outperform the simpler linear model on this dataset.

---

## Conclusion

This project successfully developed and evaluated machine learning models for house price prediction. After preprocessing the housing dataset and training multiple regression models, **Linear Regression** produced the best results with an RMSE of **1.32 million** and an R² score of **0.6529**.

The results indicate that the relationship between the available housing features and house prices can be effectively captured using a linear regression approach. Future improvements could include hyperparameter tuning, additional feature engineering, and testing advanced ensemble models such as Gradient Boosting or XGBoost.
