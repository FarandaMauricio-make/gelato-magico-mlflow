import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib

DATA_PATH = "data/ice_cream_sales.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[["temperature_c"]]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("gelato-magico-sales")

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # salva modelo local
        joblib.dump(model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)

        # registra no MLflow como modelo
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("Treino concluído!")
        print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")

if __name__ == "__main__":
    main()
