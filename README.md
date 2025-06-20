# 📊 Customer Churn Prediction System

This is a Machine Learning–powered Streamlit web application that predicts whether a telecom customer is likely to churn based on key attributes like tenure, charges, and customer demographics.

---

## 🚀 Live Demo

🔗 [Click to Try the App](https://customer-churn-prediction-ergnyegontmq6nbxuuykcj.streamlit.app)



---

## 🎯 Features

- 🔍 Predicts customer churn using **Logistic Regression**
- 📥 Easy-to-use form to input customer details
- 🧠 Displays rule-based reasoning behind predictions (e.g., "low tenure", "no dependents")
- 📊 Shows **feature importance** using model coefficients
- 📈 Interactive data visualizations including:
  - Bar chart (churn distribution)
  - KDE plot (monthly charges)
  - Stacked histogram (tenure vs churn)
  - Pie chart (contract types)
  - Heatmap (correlation matrix)
- ✅ Simple to run locally using Streamlit

---

## 🛠️ Tech Stack

| Layer        | Tools/Libraries                     |
|--------------|-------------------------------------|
| UI           | [Streamlit](https://streamlit.io)   |
| ML Model     | Logistic Regression (scikit-learn)  |
| Data Handling| Pandas, NumPy                       |
| Visuals      | Seaborn, Matplotlib                 |
| Model Saving | Joblib                              |

---

## 📁 Folder Structure

CustomerChurnPrediction/
├── app.py # Main Streamlit app
├── train_model.py # ML model training script
├── churn_model.pkl # Trained model used for prediction
├── Telco-Customer-Churn.csv # Dataset (optional, ignored in repo)
├── .gitignore
├── README.md
└── images/ # Screenshots used in README


---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Praveena23-2003/Customer-Churn-Prediction.git
cd CustomerChurnPrediction

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install required libraries
pip install -r requirements.txt

# 4. Train the model (only once)
python train_model.py

# 5. Run the Streamlit app
streamlit run app.py

📸 Screenshots

![Output](https://github.com/user-attachments/assets/d551fa5d-e12e-446e-8664-09e31bebb53d)
![Output1](https://github.com/user-attachments/assets/896bf0a3-0c41-4e53-8e7e-01946e2681bb)
![Output2](https://github.com/user-attachments/assets/0c420840-57a0-4246-920e-fdd99485ac9c)
![Output3](https://github.com/user-attachments/assets/37688d0e-40a2-48f3-91d5-00561a1c5e40)
![Output4](https://github.com/user-attachments/assets/0be0c4f2-2172-42b3-8f42-bc4ac9e69f71)
![Output5](https://github.com/user-attachments/assets/5da3b7a3-495b-4962-8509-1df356b3bb0a)
![Output6](https://github.com/user-attachments/assets/0fe700d9-4ba7-4af2-a089-8b7be4897fc1)
![Output7](https://github.com/user-attachments/assets/51c3b9e1-e40f-4c6f-aac3-bc93a5aec3ee)

📚 Dataset Source
📂 IBM Telco Customer Churn Dataset

👩‍💻 Developed By
Praveena R
🎓 MCA Student | AI & Data Enthusiast
🔗 GitHub: https://github.com/Praveena23-2003
🔗 LinkedIn: https://www.linkedin.com/in/praveena-r-5b733a237?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app









