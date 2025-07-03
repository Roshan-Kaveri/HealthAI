
import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from mlxtend.classifier import StackingClassifier
from xgboost import XGBClassifier

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.median())
    df = df.astype('float64')
    x = df.drop('Clinical progression', axis=1)
    y = df['Clinical progression'].astype(int)
    return train_test_split(x, y, test_size=0.2, random_state=42)

def scale_and_select_features(x_train, y_train, x_test, k=15):
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    x_train_sel = selector.fit_transform(x_train_scaled, y_train)
    x_test_sel = selector.transform(x_test_scaled)

    top_features = selector.get_support(indices=True)
    feature_names = x_train.columns[top_features]
    return pd.DataFrame(x_train_sel, columns=feature_names), pd.DataFrame(x_test_sel, columns=feature_names), feature_names, scaler

def train_model(model, param_grid, x_train, y_train):
    search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3, n_jobs=-1, verbose=0)
    search.fit(x_train, y_train)
    return search

def export_models(models, scaler, feature_names, x_train, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, f"{output_dir}/{name}.pkl")
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")

    medians = dict(x_train.median())
    with open(f"{output_dir}/feature_info.json", "w") as f:
        json.dump({ "features": list(feature_names), "averages": medians }, f)

def main():
    x_train, x_test, y_train, y_test = load_and_prepare_data("BD_Final.csv")

    x_train_sel, x_test_sel, feature_names, scaler = scale_and_select_features(x_train, y_train, x_test)

    models = {}

    models['random_forest'] = train_model(
        RandomForestClassifier(),
        {
            'n_estimators': [100, 300],
            'max_depth': [10, 50],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 5],
        },
        x_train_sel, y_train
    )

    models['logistic'] = train_model(
        LogisticRegression(max_iter=5000),
        {
            'C': [0.01, 0.1, 1, 10]
        },
        x_train_sel, y_train
    )

    models['knn'] = train_model(
        KNeighborsClassifier(),
        {
            'n_neighbors': list(range(3, 15))
        },
        x_train_sel, y_train
    )

    models['xgb'] = train_model(
        XGBClassifier(),
        {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        x_train_sel, y_train
    )

    models['adaboost'] = train_model(
        AdaBoostClassifier(),
        {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 1]
        },
        x_train_sel, y_train
    )

    # Stacking using base models
    stack_clf = StackingClassifier(
        classifiers=[models['random_forest'], models['logistic'], models['adaboost'], models['xgb'], models['knn']],
        use_probas=True,
        average_probas=False,
        meta_classifier=LogisticRegression()
    )
    stack_clf.fit(x_train_sel, y_train)
    models['stacking'] = stack_clf

    export_models(models, scaler, feature_names, x_train[feature_names])

    print("✅ All models and scaler exported to /models")
    print("📦 Use feature_info.json and *.pkl files in your Flask app.")

if __name__ == '__main__':
    main()
