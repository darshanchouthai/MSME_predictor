from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import os

# === Load Pretrained Model and Transformers ===
MODEL_DIR = "models"
imputer = joblib.load(os.path.join(MODEL_DIR, "imputer.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
model = joblib.load(os.path.join(MODEL_DIR, "msme_survival_model.joblib"))

# === Initialize Flask App ===
app = Flask(__name__)

# === Helper: Generate Recommendations ===
def generate_recommendations(row, feature_means, feature_names):
    recommendations = []
    for i, value in enumerate(row):
        feature = feature_names[i]
        if value < feature_means[i]:
            recommendations.append({
                "feature": feature,
                "value": round(value, 2),
                "suggestion": f"Consider improving {feature} (currently below average)."
            })
    return recommendations

# === Routes ===
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    recommendations = None
    avg_score = None

    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return render_template("index.html", error="Please upload a valid CSV file.")
        
        user_df = pd.read_csv(file)

        # ✅ Require at least 6 columns
        if user_df.shape[1] < 6:
            return render_template("index.html", error="Please upload data with at least 6 columns.")

        # Preprocess
        user_imputed = imputer.transform(user_df)
        user_scaled = scaler.transform(user_imputed)

        # Predict probabilities
        probabilities = model.predict_proba(user_scaled)[:, 1]  # Bankruptcy prob
        avg_score = 1 - np.mean(probabilities)  # Survival score

        # Recommendations
        feature_names = user_df.columns
        feature_means = np.mean(imputer.transform(np.ones((1, len(feature_names)))), axis=0)

        recommendations = []
        for i, row in enumerate(user_imputed):
            recs = generate_recommendations(row, feature_means, feature_names)
            recommendations.append({
                "row": i + 1,
                "issues": recs
            })

        prediction = f"Average Survival Score: {avg_score:.2f}"

    return render_template("index.html", prediction=prediction, recommendations=recommendations)

# === Run Server ===
if __name__ == "__main__":
    app.run(debug=True)
