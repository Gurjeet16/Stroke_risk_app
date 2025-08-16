# 🩺 Stroke Risk Prediction Web App

A **Flask-based web application** that predicts the risk of stroke based on health parameters.  
The model is trained on an Indian population dataset and provides **personalized health recommendations** along with the risk score.

---

## 📖 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Example Prediction Flow](#example-prediction-flow)
- [Screenshots](#screenshots)
- [License](#license)

---

## 📌 Overview
This project predicts the probability of an individual having a **stroke** using demographic and health-related features.  
It uses a **pre-trained machine learning model** stored in `rmodel6.pkl` with a **custom sigmoid calibration** to convert raw model probabilities into a meaningful risk percentage.

---

## ✨ Features
- Web-based **interactive form** to input health details.
- **Automatic feature engineering**:
  - Calculates `Cardiac_Stress_Score` from hypertension, heart disease, and stress levels.
- **Input validation**:
  - Age must be 18–100
  - Glucose level must be 50–400
  - BMI must be 15–50
  - Binary fields (0/1) enforced
- Risk levels:
  - **Low Risk** (< 20%)
  - **Moderate Risk** (20–50%)
  - **High Risk** (> 50%)
- Health suggestions based on risk category.

---

## 📊 Dataset
- **Name:** `Stroke_Prediction_Indians.csv`
- **Source:** Indian population health dataset.
- **Target:** Stroke occurrence.
- **Features include:**
  - Age
  - Hypertension (0/1)
  - Heart Disease (0/1)
  - Average Glucose Level
  - Chronic Stress (0/1)
  - BMI
  - Smoking Status (0/1)
  - Physical Activity (0/1)

---

## 🧠 Model Details
- **Algorithm:** Scikit-learn classifier (saved as `rmodel6.pkl`)
- **Scaler:** StandardScaler (`scaler6.pkl`)
- **Encoders:** LabelEncoders for categorical features (`label_encoders6.pkl`)
- **Calibration:** Custom sigmoid function with parameters `(a=4.5, b=-2)` for better probability mapping.

---

## 🛠 Tech Stack
- **Backend:** Python 3, Flask
- **Machine Learning:** Scikit-learn, NumPy, Pandas
- **Frontend:** HTML (Jinja2 templates), Bootstrap (optional)
- **Deployment-ready** for platforms like Heroku, Render, or local server.

---

## 📂 Project Structure
```
.
├── app6.py                     # Flask application
├── rmodel6.pkl                 # Trained ML model
├── scaler6.pkl                 # Scaler for preprocessing
├── label_encoders6.pkl         # Encoders for categorical variables
├── stroke_pred6.ipynb          # Jupyter Notebook for training
├── Stroke_Prediction_Indians.csv # Dataset
├── templates/
│   └── index6.html             # HTML UI template
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<USERNAME>/<REPO>.git
cd <REPO>
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Mac/Linux
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run the Flask app:
```bash
python app6.py
```

### Access the web interface:
- Open **http://127.0.0.1:5000/** in your browser.

---

## 🔄 Example Prediction Flow
1. Open the web app.
2. Fill in the health details:
   - Age: `45`
   - Hypertension: `1`
   - Heart Disease: `0`
   - Average Glucose Level: `120`
   - Chronic Stress: `0`
   - BMI: `27`
   - Smoking Status: `1`
   - Physical Activity: `1`
3. Click **Predict**.
4. Get:
   - **Risk Score:** `42.15%`
   - **Risk Level:** Moderate Risk
   - **Suggestions:** Personalized health tips.

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
