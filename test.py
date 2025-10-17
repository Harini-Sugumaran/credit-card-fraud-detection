import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# 1. Load dataset
df = pd.read_csv("creditcard.csv")

# 2. Select ONLY 6 features
X = df[["Time", "Amount", "V1", "V2", "V3", "V4"]]
y = df["Class"]
print("Training features shape:", X.shape)  # check → should be (rows, 6)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Logistic Regression
model = LogisticRegression(max_iter=500, class_weight="balanced")
model.fit(X_train_scaled, y_train)

# 6. Save model & scaler
pickle.dump(model, open("fraud_model_6.pkl", "wb"))
pickle.dump(scaler, open("scaler_6.pkl", "wb"))
print("✅ Saved fraud_model_6.pkl and scaler_6.pkl")


# 7. Evaluatetest
y_pred = model.predict(X_test_scaled)
print("\n✅ Model Training Complete!\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
