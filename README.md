# 📉 Customer Churn Prediction — End-to-End Machine Learning Project
https://churnproject-p1.streamlit.app/

Predict whether a telecommunications customer is likely to churn using machine learning, FastAPI, and Streamlit.

This project demonstrates a complete production-style data science pipeline:

* 🧹 Data preprocessing

* 📊 Exploratory data analysis

* 🤖 Model training & evaluation

* 🔧 Model packaging using joblib

* 🚀 Serving predictions using FastAPI or Streamlit

* 🌐 Deploying an interactive web UI

## 🔍 1. Project Overview

Customer churn is a major problem for subscription-based businesses. Retaining existing customers is far cheaper than acquiring new ones.

This project uses the popular Telco Customer Churn dataset to build a predictive model that estimates:

“How likely is this customer to leave (churn)?”

The final app outputs both:

Churn Probability (%)

Churn Prediction (Yes/No)

## 🧠 2. Features Used

Below are the key input features passed into the model and app:

*Customer Demographics
*Feature	Description
*gender	Male/Female
*SeniorCitizen	1 = Yes, 0 = No
*Partner	Whether customer has a partner
*Dependents	Whether customer has dependents
*Subscription Attributes
*Feature	Description
*tenure	Number of months with the company
*Contract	Month-to-month / One year / Two year
*PaperlessBilling	Yes/No
*PaymentMethod	eCheck, Credit card, Bank transfer, etc.
*Services Subscribed
*Feature	Description
*PhoneService	Yes/No
*MultipleLines	Yes/No/No phone service
*InternetService	Fiber, DSL, None
*OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport	Optional add-on services
*StreamingTV, StreamingMovies	Entertainment options
*Financial Metrics
*Feature	Description
*MonthlyCharges	Monthly bill cost
+TotalCharges	Total spent by customer

These features were encoded using OneHotEncoder and scaled using StandardScaler inside a single Scikit-Learn pipeline.

## 🤖 3. Machine Learning Model
Chosen Model: Logistic Regression

Reasoning:

* Performs well on linearly separable churn data

* Provides interpretable coefficients

* Fast to train and deploy

* Works well with imbalanced datasets

* Pipeline Steps

* Handle missing values

* Encode categorical features (OneHotEncoder)

* Scale numerical features (StandardScaler)

* Fit Logistic Regression model

Save final pipeline as:

* models/churn_pipeline.pkl

* Model Performance (example)

ROC-AUC: ~0.82

* Accuracy: ~0.79

Precision/Recall: Balanced for churn detection

## 🏗️ 4. Tech Stack
Backend / Model Serving


* Joblib (for model serialization)

* FastAPI (optional API deployment)(Locally)

Frontend

* Streamlit (final deployment)

* Clean UI for entering customer data

* Displays churn probability and prediction cards

DevOps / Deployment

* GitHub (version control)

* Streamlit Cloud (hosted UI)
