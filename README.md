# 🩺 Medical Insurance Cost Prediction

A machine learning project to predict individual medical insurance costs based on health and demographic features. This end-to-end solution involves data preprocessing, exploratory data analysis (EDA), model training with multiple regression algorithms, MLflow tracking, and a Streamlit web application for real-time predictions.

---

## 📌 Project Overview

- **Domain**: Healthcare & Insurance
- **Goal**: Predict medical insurance charges using personal features such as age, sex, BMI, smoking status, number of children, and region.
- **Deliverables**:
  - Cleaned dataset
  - EDA insights and visualizations
  - Trained regression models
  - Streamlit web application
  - MLflow model tracking
  - Complete documentation

---

## 🛠️ Tech Stack

- **Programming Language**: Python
- **Libraries**: pandas, scikit-learn, seaborn, matplotlib, xgboost, joblib, mlflow, streamlit
- **Tools**: Jupyter Notebook, Streamlit, MLflow

---

## 📊 Dataset

**File**: `medical_insurance.csv`  
**Features**:
- `age`: Age of the individual
- `sex`: Gender (male/female)
- `bmi`: Body Mass Index
- `children`: Number of dependents
- `smoker`: Smoking status (yes/no)
- `region`: Residential area
- `charges`: Target variable (insurance cost)

---

    medical_insurance_project/
      │
      ├── data/
      │   └── medical_insurance.csv
      │
      ├── notebooks/
      │   └── EDA_Medical_Insurance.ipynb
      │
      ├── src/
      │   ├── preprocess.py
      │   ├── train_models.py
      │   └── predict_utils.py
      │
      ├── app/
      │   └── streamlit_app.py
      │
      ├── models/
      │   └── best_model.pkl
      │
      ├── mlruns/
      │   └── MLflow logs and artifacts
      │
      ├── requirements.txt
      └── README.md

---

## 🔍 Exploratory Data Analysis

- Distribution of charges, age, and BMI
- Count of smokers vs non-smokers
- Region-wise distribution
- Bivariate analysis: Age vs Charges, Smoker vs Charges
- Correlation heatmap of numerical features

---

## 🤖 Models Trained

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor

📌 **Evaluation Metrics**:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

✅ The best model is saved in `models/best_model.pkl` and logged in MLflow.

---

## 🌐 Streamlit Web App

Launch an interactive web interface for:
- Visualizing EDA insights
- Entering user details (age, gender, BMI, etc.)
- Predicting medical insurance costs in real-time

🖥️ Run with:
```bash
streamlit run app/streamlit_app.py
