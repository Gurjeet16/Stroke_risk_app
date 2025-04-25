from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

model = joblib.load('rmodel6.pkl')
scaler = joblib.load('scaler6.pkl')
label_encoders = joblib.load('label_encoders6.pkl')

# Custom sigmoid calibration
def custom_sigmoid(probs, a=4.5, b=-2):
    return 1 / (1 + np.exp(-(a * probs + b)))

feature_order = ['Age', 'Hypertension', 'Heart Disease', 'Average Glucose Level', 'Chronic Stress', 'BMI', 'Smoking Status', 'Physical Activity', 'Cardiac_Stress_Score']

@app.route('/')
def home():
    return render_template('index6.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
        # Input validation
        if not (18 <= input_data['Age'] <= 100):
            return render_template('index6.html', prediction_text="Error: Age must be between 18 and 100")
        if not (50 <= input_data['Average Glucose Level'] <= 400):
            return render_template('index6.html', prediction_text="Error: Glucose Level must be between 50 and 400")
        if not (15 <= input_data['BMI'] <= 50):
            return render_template('index6.html', prediction_text="Error: BMI must be between 15 and 50")
        for field in ['Hypertension', 'Heart Disease', 'Chronic Stress', 'Smoking Status', 'Physical Activity']:
            if input_data[field] not in [0, 1]:
                return render_template('index6.html', prediction_text=f"Error: {field} must be 0 or 1")
        
        input_df = pd.DataFrame([input_data])
        input_df['Cardiac_Stress_Score'] = (input_df['Heart Disease'] * 0.4 + input_df['Hypertension'] * 0.4 + input_df['Chronic Stress'] * 0.2) * 100
        input_df = input_df[feature_order]
        input_scaled = scaler.transform(input_df).astype(np.float32)
        prob = model.predict_proba(input_scaled)[0, 1]
        risk_score = custom_sigmoid(prob) * 100
        # Risk interpretation and suggestions
        if risk_score < 20:
            risk_level = "Low Risk"
            suggestions = [
                "Continue regular physical activity (e.g., 150 min/week moderate exercise).",
                "Maintain a balanced diet low in sodium and trans fats.",
                "Monitor blood pressure and glucose annually.",
                "Avoid smoking or secondhand smoke exposure."
            ]
        elif risk_score <= 50:
            risk_level = "Moderate Risk"
            suggestions = [
                "Consult a doctor for cardiovascular screening (e.g., blood pressure, cholesterol).",
                "Increase physical activity if low (aim for 30 min/day, 5 days/week).",
                "Reduce stress through mindfulness or therapy.",
                "Cut back on smoking or enroll in a cessation program."
            ]
        else:
            risk_level = "High Risk"
            suggestions = [
                "Seek a doctor's advice urgently to manage risk factors (e.g., hypertension, diabetes).",
                "Follow prescribed treatments (e.g., statins, antihypertensives).",
                "Quit smoking immediately with professional support.",
                "Adopt a heart-healthy diet (e.g., DASH diet) and exercise under medical guidance."
            ]
        return render_template('index6.html', 
                             prediction_text=f'Risk Score: {risk_score:.2f}% ({risk_level})',
                             suggestions=suggestions)
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return render_template('index6.html', prediction_text=f"Error: Invalid input - {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)