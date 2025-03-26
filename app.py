# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('insurance_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])
        
        # Create feature array
        features = np.array([[age, sex, bmi, children, smoker, region]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return render_template('output.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)