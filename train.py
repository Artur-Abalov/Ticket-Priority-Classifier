
import sqlite3

import joblib
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


def load_training_data():
    conn = sqlite3.connect('tickets.sqlite')
    try:
        df = pd.read_sql_query("SELECT * FROM tickets", conn)
        # drop noise
        encoded_df = df.drop(
            columns=[
                "ticket_id",
                "day_of_week",
                "company_id",
                "company_size",
                "industry",
                "customer_tier",
                "region",
                "product_area",
                "booking_channel",
                "reported_by_role",
                "customer_sentiment",
                "priority",
            ]
        )
        X = encoded_df.drop(columns=["priority_cat"])
        y = encoded_df["priority_cat"]

        return X, y
    finally:
        conn.close()


def train():
    X, y = load_training_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    base_kwargs = {
        "objective": "multi:softmax",
        "learning_rate": 0.01,
        "tree_method": "hist",
        "n_estimators": 1000,
        "max_depth": 6,
        "subsample": 0.8,
        "reg_alpha": 3,
        "min_child_weight": 3,
    }
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    old_model = joblib.load("pipeline.joblib")
    model = xgb.XGBClassifier(**base_kwargs)
    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        model,
        param_distributions={
            "n_estimators": randint(400, 2000),
            "max_depth": randint(3, 11),
            "learning_rate": loguniform(1e-3, 3e-1),
            "min_child_weight": loguniform(1, 16),

            "subsample": uniform(0.55, 0.45),
            "colsample_bytree": uniform(0.55, 0.45),
            "colsample_bynode": uniform(0.5, 0.5),

            "gamma": loguniform(1e-4, 10),
            "reg_lambda": loguniform(1e-3, 1e3),
            "reg_alpha": loguniform(1e-4, 10),

            "max_bin": randint(128, 512),

            "grow_policy": ["depthwise", "lossguide"],
            "max_leaves": randint(16, 512),
            
        },
        cv=k_fold,
        n_iter=20,
        scoring='f1_macro',
        error_score='raise',
        verbose=1,
        random_state=42
    )

    search.fit(X_train, y_train_encoded)

    best_estimator = search.best_estimator_
    predictions = best_estimator.predict(X_test)
    old_predictions = old_model.predict(X_test)

    score = f1_score(y_test_encoded, predictions, average="macro")
    old_score = f1_score(y_test_encoded, old_predictions, average="macro")
    if score >= old_score:
        joblib.dump(best_estimator, "pipeline.joblib")
    else:
        print("trained a worse model, not saving it")

if __name__ == "__main__":
    train()
