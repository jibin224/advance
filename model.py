# model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load and preprocess data
insurance_dataset = pd.read_csv('C:/Users/jibin/OneDrive/Desktop/jp/insurance.csv')
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)

# Split features and target
X = insurance_dataset.drop(columns='charges', axis=1)
y = insurance_dataset['charges']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open('insurance_model.pkl', 'wb') as file:
    pickle.dump(model, file)