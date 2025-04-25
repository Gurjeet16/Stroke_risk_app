from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load('rmodel4.pkl')
scaler = joblib.load('scaler4.pkl')
label_encoders = joblib.load('label_encoders4.pkl')

# Custom sigmoid calibration
def custom_sigmoid(probs, a=4, b=-1.5):
    return 1 / (1 + np.exp(-(a * probs + b)))

feature_order = ['Age', 'Hypertension', 'Heart Disease', 'Average Glucose Level', 'Chronic Stress', 'BMI', 'Smoking Status', 'Physical Activity', 'Cardiac_Stress_Score']

@app.route('/')
def home():
    return render_template('index5.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'Age': float(request.form['Age']),
        'Hypertension': int(request.form['Hypertension']),
        'Heart Disease': int(request.form['Heart Disease']),
        'Average Glucose Level': float(request.form['Average Glucose Level']),
        'Chronic Stress': int(request.form['Chronic Stress']),
        'BMI': float(request.form['BMI']),
        'Smoking Status': int(request.form['Smoking Status']),
        'Physical Activity': int(request.form['Physical Activity'])
    }
    input_df = pd.DataFrame([input_data])
    # Compute Cardiac_Stress_Score
    input_df['Cardiac_Stress_Score'] = (input_df['Heart Disease'] * 0.4 + input_df['Hypertension'] * 0.4 + input_df['Chronic Stress'] * 0.2) * 100
    input_df = input_df[feature_order]
    input_scaled = scaler.transform(input_df).astype(np.float32)
    prob = model.predict_proba(input_scaled)[0, 1]
    risk_score = custom_sigmoid(prob) * 100
    # Risk interpretation
    if risk_score < 20:
        risk_level = "Low Risk"
    elif risk_score <= 50:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"
    return render_template('index5.html', prediction_text=f'Risk Score: {risk_score:.2f}% ({risk_level})')

if __name__ == '__main__':
    app.run(debug=True)