import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, url_for, redirect
import joblib, pandas as pd, numpy as np, os, json, shap, uuid
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

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

# ── Load models and data ───────────────────────
models = {
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "Logistic":      joblib.load("models/logistic.pkl"),
    "KNN":           joblib.load("models/knn.pkl"),
    "XGBoost":       joblib.load("models/xgb.pkl"),
    "AdaBoost":      joblib.load("models/adaboost.pkl"),
    "Stacking":      joblib.load("models/stacking.pkl")
}
scaler = joblib.load("models/scaler.pkl")

# Ensure directories exist
os.makedirs("static", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

with open("models/feature_info.json") as f:
    meta           = json.load(f)
    top_features   = meta["top_features"]
    medians        = meta["all_medians"]
    all_features   = meta["all_features"]
    extra_features = [f for f in all_features if f not in top_features]

X_test        = pd.read_csv("models/X_test.csv")[top_features]
y_test        = pd.read_csv("models/y_test.csv")["Clinical progression"].values
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=top_features)

# ────────────────────────── Routes ──────────────────────────
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

    # Collect input
    user_input = {f: float(request.form.get(f) or medians[f]) for f in top_features}
    X_user = pd.DataFrame(scaler.transform(pd.DataFrame([user_input])[top_features]),
                          columns=top_features)

    results = {}
    session_id = uuid.uuid4().hex

    for name, mdl in models.items():
        model = mdl.best_estimator_ if hasattr(mdl, "best_estimator_") else mdl
        y_pred = int(model.predict(X_user)[0])
        y_test_pred = model.predict(X_test_scaled)

        acc = round(accuracy_score(y_test, y_test_pred), 3)
        rec = round(recall_score(y_test, y_test_pred), 3)
        f1  = round(f1_score(y_test, y_test_pred), 3)
        cm  = confusion_matrix(y_test, y_test_pred)

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

        proba1 = lambda X: model.predict_proba(X)[:, 1]
        exp    = shap.Explainer(proba1, X_test_scaled, feature_names=top_features)
        sv     = exp(X_user)
        shap_file = f"static/shap_{uuid.uuid4().hex}.png"
        plt.figure()
        shap.plots.waterfall(sv[0], show=False, max_display=10)
        plt.savefig(shap_file, bbox_inches="tight")
        plt.close()

        results[name] = {
            "prediction": y_pred,
            "accuracy":   acc,
            "recall":     rec,
            "f1":         f1,
            "cm_image":   os.path.basename(cm_file),
            "shap_image": os.path.basename(shap_file),
            "result":     "Normal" if y_pred == 0 else "Abnormal",
        }

    # Save to file
    with open(f"results/{session_id}.json", "w") as f:
        json.dump({
            "user_input": user_input,
            "prediction_results": results
        }, f)

    return redirect(url_for("view_result", session_id=session_id))


@app.route("/result/<session_id>")
def view_result(session_id):
    try:
        with open(f"results/{session_id}.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        return "Invalid or expired session ID", 404

    return render_template("result.html",
                           user_input=data["user_input"],
                           prediction_results=data["prediction_results"])


@app.route("/predict/social-anxiety")
def predict_social_anxiety():
    return "Coming Soon: Social Anxiety Predictor."


if __name__ == "__main__":
    app.run(debug=True, port=8000)

