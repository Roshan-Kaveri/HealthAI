from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os
import json
import shap
import matplotlib.pyplot as plt
import uuid

app = Flask(__name__)

# Load models
models = {
    'Random Forest': joblib.load('models/random_forest.pkl'),
    'Logistic Regression': joblib.load('models/logistic.pkl'),
    'KNN': joblib.load('models/knn.pkl'),
    'XGBoost': joblib.load('models/xgb.pkl'),
    'AdaBoost': joblib.load('models/adaboost.pkl'),
    'Stacking': joblib.load('models/stacking.pkl')
}

scaler = joblib.load('models/scaler.pkl')

# Load top 15 features and their medians
with open("models/feature_info.json", "r") as f:
    info = json.load(f)
    top_features = info['features']
    median_values = info['averages']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/respiratory-disease-child', methods=['GET', 'POST'])
def predict_disease():
    prediction_results = {}
    user_input = {}
    xai_images = {}

    if request.method == 'POST':
        for feature in top_features:
            val = request.form.get(feature)
            user_input[feature] = float(val) if val else median_values[feature]

        input_df = pd.DataFrame([user_input])
        scaled_input = scaler.transform(input_df)

        for name, model in models.items():
            y_pred = model.predict(scaled_input)[0]
            y_proba = model.predict_proba(scaled_input)[0][1] if hasattr(model, 'predict_proba') else 'N/A'

            # SHAP
            explainer = shap.Explainer(model.predict, pd.DataFrame([user_input]))
            shap_values = explainer(pd.DataFrame([user_input]))
            filename = f'static/shap_{uuid.uuid4().hex}.png'
            plt.figure()
            shap.plots.waterfall(shap_values[0], show=False)
            plt.savefig(os.path.join('static', os.path.basename(filename)), bbox_inches='tight')
            plt.close()

            prediction_results[name] = {
                'prediction': y_pred,
                'probability': y_proba,
                'xai_image': filename
            }

        return render_template('result.html', user_input=user_input, prediction_results=prediction_results)

    return render_template('predict_form.html', features=top_features, medians=median_values)

@app.route('/predict/social-anxiety')
def predict_social_anxiety():
    return "Coming Soon: Social Anxiety Predictor."

if __name__ == '__main__':
    app.run(debug=True)