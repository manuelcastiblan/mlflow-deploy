# src/train.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import mlflow
import os

# Simulamos un dataset
df = pd.DataFrame({
    "x": range(100),
    "y": [2*i + 3 + (i % 5) for i in range(100)]
})

X = df[["x"]]
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

import os

if os.getenv("GITHUB_ACTIONS") == "true":
    # Tracking relativo para GitHub Actions
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
else:
    # Ruta fija para uso local
    mlflow.set_tracking_uri("file:///home/manuelcastiblan/academic/mlflow-deploy/mlflow-deploy/mlruns")
mlflow.set_experiment("CI-CD-Lab")



with mlflow.start_run():
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse", mse)
    joblib.dump(model, "model.pkl")
    mlflow.sklearn.log_model(model, "model", registered_model_name="ci-cd-model")

print(f"✅ Entrenamiento completo. MSE: {mse:.4f}")
