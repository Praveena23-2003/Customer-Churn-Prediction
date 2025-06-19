import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

print("📥 Loading dataset...")

# Step 1: Load dataset
if not os.path.exists("Telco-Customer-Churn.csv"):
    print("❌ Dataset not found! Make sure 'Telco-Customer-Churn.csv' is in the same folder.")
    exit()

df = pd.read_csv("Telco-Customer-Churn.csv")
print("✅ Dataset loaded.")

# Step 2: Clean and preprocess
print("🧹 Cleaning data...")
df.dropna(inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

categorical_cols = df.select_dtypes(include='object').columns.tolist()
if 'Churn' in categorical_cols:
    categorical_cols.remove('Churn')


le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("✅ Data preprocessing done.")

# Step 3: Train model
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("⚙️ Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# Step 4: Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Save the model
joblib.dump(model, "churn_model.pkl")
print("✅ Model saved as churn_model.pkl 🎉")
