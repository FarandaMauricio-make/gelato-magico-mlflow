import joblib
import pandas as pd

MODEL_PATH = "models/model.joblib"

def predict(temp_c: float) -> float:
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame({"temperature_c": [temp_c]})
    return float(model.predict(X)[0])

if __name__ == "__main__":
    temp = float(input().strip())
    sales = predict(temp)
    print(f"Previsão de vendas para {temp}°C: {sales:.0f} sorvetes")
