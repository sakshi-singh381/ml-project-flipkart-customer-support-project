import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- LOAD DATA ----------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Customer_support_data (1).csv")

df = pd.read_csv(DATA_PATH)

# clean columns
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

# ---------------- CSAT LABEL CREATION ----------------
def csat_group(score):
    if score <= 2:
        return "Low"
    elif score == 3:
        return "Neutral"
    else:
        return "High"

df["csat_label"] = df["csat_score"].apply(csat_group)

# ---------------- FEATURES ----------------
features = [
    "category",
    "sub-category",
    "customer_city",
    "product_category",
    "connected_handling_time",
    "agent_shift",
    "tenure_bucket"
]

df = df[features + ["csat_label"]]

# ---------------- ENCODING ----------------
encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ---------------- SPLIT ----------------
X = df.drop("csat_label", axis=1)
y = df["csat_label"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ---------------- MODEL ----------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------- PREDICTION ----------------
y_pred = model.predict(X_test)

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low", "Neutral", "High"],
    yticklabels=["Low", "Neutral", "High"]
)

plt.title("CSAT Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

os.makedirs("output", exist_ok=True)
plt.savefig("output/confusion_matrix.png")

# ---------------- REPORT ----------------
print("\n📊 Classification Report:\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Low", "Neutral", "High"]
))

# ---------------- SAVE MODEL ----------------
os.makedirs("ml_model", exist_ok=True)

joblib.dump(model, "ml_model/csat_model.pkl")
joblib.dump(encoders, "ml_model/csat_encoders.pkl")
joblib.dump(label_encoder, "ml_model/csat_label_encoder.pkl")

print("\n✅ MODEL TRAINED SUCCESSFULLY")
print("📁 Files saved in ml_model/ and output/")