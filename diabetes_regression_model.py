import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Load the dataset
url = '/processed_diabetes.csv'  # Update this path to your dataset
data = pd.read_csv(url)

# Display the first few rows of the dataset
print(data.head())

# Preprocessing
# Check for missing values
print(data.isnull().sum())

# Define the feature variables (X) and the target variable (y)
# Assuming 'diabetes' is the target variable (1 for positive, 0 for negative)
X = data.drop(columns=['Diabetes_012'])  # Drop the target variable from features
y = data['Diabetes_012']

# --- Changes here ---
# Convert y to binary (0 or 1) if it contains other values like 2
y = y.map({0: 0, 1: 1, 2:1}) # Assuming 2 represents a positive case like 1
# --- End of changes ---

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Optional: Using statsmodels for detailed statistics
X_train_sm = sm.add_constant(X_train)  # Add a constant for the intercept
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()

# Print the summary of the logistic regression model
print(result.summary())

#
