# 📊 Customer Churn Prediction System

This is a Machine Learning–powered **Streamlit web app** that predicts whether a telecom customer is likely to churn based on key input details like tenure, charges, and demographics.

---

## 🚀 Live Demo

🔗 [Click to Try the App](http://localhost:8501)

> Replace this with your actual Streamlit app URL after deployment.

---

## 🎯 Features

- 🔍 Predicts customer churn based on logistic regression
- 📥 Simple input form for entering customer details
- 🧠 Rule-based explanation for churn (e.g., "low tenure", "no partner")
- 📊 ML-based feature importance (coefficients)
- 📈 Rich data visualizations (bar chart, KDE, pie chart, heatmap)
- ✅ Easy to run locally with Streamlit

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python-based UI)
- **Model**: Logistic Regression (from scikit-learn)
- **Backend**: Python, Pandas, Joblib
- **Visualization**: Seaborn, Matplotlib

---

## 📁 Folder Structure

CustomerChurnPrediction/
├── app.py # Streamlit app
├── train_model.py # Model training script
├── churn_model.pkl # Trained model (used by app)
├── Telco-Customer-Churn.csv (ignored)
├── .gitignore
└── README.md


---

## ⚙️ How to Run Locally

# Clone the repo
git clone https://github.com/yourusername/CustomerChurnPrediction.git
cd CustomerChurnPrediction

# (Optional) create virtual env
# python -m venv venv && source venv/bin/activate

# Install required packages
pip install -r requirements.txt  # (if using)

# Train model (only once)
python train_model.py

# Run the app
streamlit run app.py

📸 Screenshots





