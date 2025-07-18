import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
import os
import warnings


def train_all_models(df):
    X = df.drop("charges", axis=1)
    y = df["charges"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    best_r2 = float("-inf")
    best_model = None
    best_name = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Safely calculate metrics
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                try:
                    r2 = r2_score(y_test, y_pred)
                except:
                    r2 = float("-inf")

            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)

            # Log model
            input_example = X_test.iloc[:1]
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
            )

            print(f"[{name}] R2: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

            if np.isfinite(r2) and r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name = name

    # Fallback: Save last trained model if no R2 > -inf
    if best_model:
        joblib.dump(best_model, "models/best_model.pkl")
        print(f"\n✅ Best model saved: {best_name} (R² = {best_r2:.4f}) ➜ models/best_model.pkl")
    else:
        print("❌ No valid model trained.")

    return best_model


if __name__ == "__main__":
    from preprocess import load_and_preprocess

    print("🔁 Loading and preprocessing data...")
    df, _, _, _ = load_and_preprocess("data/medical_insurance.csv")

    os.makedirs("models", exist_ok=True)

    print("🚀 Training models...")
    train_all_models(df)
    print("✅ Training pipeline completed.")
