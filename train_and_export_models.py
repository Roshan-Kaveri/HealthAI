import pandas as pd
import numpy as np
import json
import os
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from mlxtend.classifier import StackingClassifier
from xgboost import XGBClassifier

excluded = ["Gentamicin", "Length of stay in intensive care unit", "Main diagnostic"]


def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.median())
    df = df.astype('float64')
    x = df.drop('Clinical progression', axis=1)
    y = df['Clinical progression'].astype(int)
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


def scale_and_select_features(x_train, x_test):
    # keep every numeric column except the ones you listed
    numeric_cols = [c for c in x_train.columns if c not in excluded]

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train[numeric_cols])
    x_test_scaled  = scaler.transform(x_test[numeric_cols])

    return (pd.DataFrame(x_train_scaled, columns=numeric_cols),
            pd.DataFrame(x_test_scaled , columns=numeric_cols),
            numeric_cols,   # ← this is feature_names
            scaler)


def train_model(model, param_grid, x_train, y_train):
    search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3, n_jobs=-1, verbose=0)
    search.fit(x_train, y_train)
    return search

def export_models(models, scaler, top_features, x_train_top, all_features, all_medians, x_test, y_test, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, f"{output_dir}/{name}.pkl")
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")

    pd.DataFrame(x_test).to_csv(f"{output_dir}/X_test.csv", index=False)
    pd.DataFrame(y_test, columns=["Clinical progression"]).to_csv(f"{output_dir}/y_test.csv", index=False)

    medians_top = dict(x_train_top.median())
    info = {
        "top_features": list(top_features),
        "top_medians": medians_top,
        "all_features": list(all_features),
        "all_medians": all_medians,
    }
    with open(f"{output_dir}/feature_info.json", "w") as f:
        json.dump(info, f)

def main():
    x_train, x_test, y_train, y_test = load_and_prepare_data("BD_Final.csv")
    x_train_sel, x_test_sel, feature_names, scaler = scale_and_select_features(x_train, x_test)

    # ── SMOTE to balance class-1 vs class-0 ───────────────────────
    sm = SMOTE(random_state=42, k_neighbors=5)
    x_train_sel, y_train = sm.fit_resample(x_train_sel, y_train)
    print("After SMOTE  →  class counts:", np.bincount(y_train))

    x_train_top = x_train[feature_names]


    all_features = [c for c in x_train.columns if c not in excluded]
    all_medians = dict(x_train[all_features].median())

    models = {}
    models['random_forest'] = train_model(
        RandomForestClassifier(),
        {'n_estimators': [100, 300], 'max_depth': [10, 50], 'min_samples_split': [2, 10], 'min_samples_leaf': [1, 5]},
        x_train_sel, y_train
    )
    models['logistic'] = train_model(
        LogisticRegression(max_iter=5000),
        {'C': [0.01, 0.1, 1, 10]},
        x_train_sel, y_train
    )
    models['knn'] = train_model(
        KNeighborsClassifier(),
        {'n_neighbors': list(range(3, 15))},
        x_train_sel, y_train
    )
    models['xgb'] = train_model(
        XGBClassifier(),
        {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]},
        x_train_sel, y_train
    )
    models['adaboost'] = train_model(
        AdaBoostClassifier(),
        {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 1]},
        x_train_sel, y_train
    )

    stack_clf = StackingClassifier(
        classifiers=[models['random_forest'], models['logistic'], models['adaboost'], models['xgb'], models['knn']],
        use_probas=True,
        average_probas=False,
        meta_classifier=LogisticRegression()
    )
    stack_clf.fit(x_train_sel, y_train)
    models['stacking'] = stack_clf

    export_models(models, scaler, feature_names, x_train_top, all_features, all_medians, x_test_sel, y_test)
    print("✅ Models & metadata saved in /models")

if __name__ == '__main__':
    main()
