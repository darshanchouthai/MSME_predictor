import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# === Config ===
DATA_PATH = "data/polish_companies_bankruptcy_complete.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# === Load Data ===
print("📦 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# === Feature and Target Separation ===
print("🔍 Preparing features and labels...")
X = df.drop(columns=["class", "year"])  # 64 features
y = df["class"]  # Target column

# === Handle Missing Values ===
print("🧼 Imputing missing values with median...")
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# === Scaling ===
print("📊 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Class Imbalance Handling for XGBoost ===
scale_pos_weight = np.bincount(y_train)[0] / np.bincount(y_train)[1]

# === Define Models ===
print("🚀 Initializing models...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=scale_pos_weight
)

# === Voting Ensemble ===
ensemble = VotingClassifier(estimators=[
    ("rf", rf_model),
    ("xgb", xgb_model)
], voting="soft")

# === Train Model ===
print("🎯 Training ensemble model...")
ensemble.fit(X_train, y_train)

# === Evaluation ===
print("📈 Evaluating model...")
y_pred = ensemble.predict(X_test)
report = classification_report(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("✅ Accuracy:", acc)
print(report)

# === Save Model and Transformers ===
print("💾 Saving model and preprocessing steps...")
joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(ensemble, os.path.join(MODEL_DIR, "msme_survival_model.joblib"))

print("✅ Training complete and models saved in 'models/' directory.")
