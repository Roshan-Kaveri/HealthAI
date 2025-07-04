# ─────────────────────────── app.py ───────────────────────────
import matplotlib
matplotlib.use("Agg")                      # 100 % head-less
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, url_for
import joblib, pandas as pd, numpy as np, os, json, shap, uuid
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, confusion_matrix
)

app = Flask(__name__)
MY_TOP_FEATURES_RAW = [
    "Kaliemia",
    "Oxygen saturation (SaO2) at admission",
    "Location of pleural effusion",
    "Weight (Kg)",
    "Patient transferred to intensive care unit",
    "If yes. which antibiotics",
    "Chest X-ray finding",
    "Natremia",
    "Number of Hib vaccine doses received",
    "Urea",
    "Heart rate",
]
# ── load models & metadata ────────────────────────────────────
models = {
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "Logistic":      joblib.load("models/logistic.pkl"),
    "KNN":           joblib.load("models/knn.pkl"),
    "XGBoost":       joblib.load("models/xgb.pkl"),
    "AdaBoost":      joblib.load("models/adaboost.pkl"),
    "Stacking":      joblib.load("models/stacking.pkl")
}
scaler = joblib.load("models/scaler.pkl")

if not os.path.isdir("static"):
    os.makedirs("static", exist_ok=True)

if not os.path.isdir("models"):
    os.makedirs("models", exist_ok=True)


with open("models/feature_info.json") as f:
    meta           = json.load(f)
    top_features   = meta["top_features"]
    medians        = meta["all_medians"]
    all_features   = meta["all_features"]
    extra_features = [f for f in all_features if f not in top_features]

X_test        = pd.read_csv("models/X_test.csv")[top_features]
y_test        = pd.read_csv("models/y_test.csv")["Clinical progression"].values
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=top_features)

# ────────────────────────── routes ────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict/respiratory-disease-child", methods=["GET", "POST"])
def predict_disease():
    if request.method != "POST":
        return render_template(
            "predict_form.html",
            top_features=MY_TOP_FEATURES_RAW,
            extra_features=[f for f in top_features if f not in MY_TOP_FEATURES_RAW],
            medians=medians,
        )

    # collect user input
    user_input = {f: float(request.form.get(f) or medians[f]) for f in top_features}
    X_user = pd.DataFrame(scaler.transform(pd.DataFrame([user_input])[top_features]),
                          columns=top_features)

    results = {}

    for name, mdl in models.items():
        model = mdl.best_estimator_ if hasattr(mdl, "best_estimator_") else mdl

        # predictions & metrics
        y_pred      = int(model.predict(X_user)[0])
        y_test_pred = model.predict(X_test_scaled)
        acc, rec, f1 = (
            round(accuracy_score(y_test, y_test_pred), 3),
            round(recall_score(y_test, y_test_pred), 3),
            round(f1_score(y_test, y_test_pred), 3),
        )
        cm = confusion_matrix(y_test, y_test_pred)

        # confusion-matrix figure
        cm_file = f"static/cm_{uuid.uuid4().hex}.png"
        plt.figure(figsize=(3, 3))
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(cm_file, dpi=110, bbox_inches="tight")
        plt.close()

        # SHAP waterfall (probability of class 1)
        proba1 = lambda X: model.predict_proba(X)[:, 1]
        exp    = shap.Explainer(proba1, X_test_scaled, feature_names=top_features)
        sv     = exp(X_user)
        shap_file = f"static/shap_{uuid.uuid4().hex}.png"
        plt.figure()

        shap.plots.waterfall(sv[0], show=False, max_display=10)

        plt.savefig(shap_file, bbox_inches="tight")
        plt.close()

        result = "Normal" if y_pred == 0 else "Abnormal"

        results[name] = {
            "prediction": y_pred,
            "accuracy":   acc,
            "recall":     rec,
            "f1":         f1,
            "cm_image":   os.path.basename(cm_file),
            "shap_image": os.path.basename(shap_file),
            "result":     result,
        }

    return render_template("result.html",
                           user_input=user_input,
                           prediction_results=results)


@app.route("/predict/social-anxiety")
def predict_social_anxiety():
    return "Coming Soon: Social Anxiety Predictor."


if __name__ == "__main__":
    app.run(debug=True)
