# Customer Churn Prediction – AI Assignment

## Project Overview
This project implements an end-to-end machine learning solution to predict customer churn.
The goal is to identify whether a customer is likely to leave a service based on demographic,
service usage, and billing information.

## Dataset
- Dataset: Telco Customer Churn Dataset
- Total rows: 7043
- Target variable: Churn
  - 0 → No Churn
  - 1 → Churn

## Setup Instructions
1. Install required libraries:

pip install flask numpy pandas scikit-learn joblib


2. Run the Flask API:

python app.py


3. Test the API:
Send a POST request to:

using Postman or curl with JSON input.

## Data Preprocessing
- Converted TotalCharges to numeric and handled missing values
- Dropped customerID column
- Encoded categorical variables using Label Encoding
- Applied feature scaling where required
- Handled class imbalance using class weights

## Model Explanation
Multiple models were trained and evaluated, including Logistic Regression,
Random Forest, and XGBoost. After evaluation and hyperparameter tuning,
Logistic Regression was selected as the final model because it achieved
the best recall and F1-score.

Recall was prioritized since identifying customers likely to churn is
more important than overall accuracy.

## Model Performance
- Accuracy: ~75%
- Recall: ~83%
- F1-score: ~0.65

## Deployment
The trained model was deployed using Flask as a REST API.
The API accepts customer data in JSON format and returns
churn predictions in real time.

## Project Structure

customer-churn-prediction/
├── app.py
├── customer_churn.ipynb
├── Telco-Customer-Churn.csv
├── models/
│ ├── churn_model.pkl
│ ├── scaler.pkl
│ ├── label_encoders.pkl
│ └── feature_columns.pkl
├── README.md


## Conclusion
This project demonstrates a complete AI workflow including
data preprocessing, model training, evaluation, hyperparameter
tuning, and deployment using Flask.

