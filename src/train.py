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

tracking_uri = "file:./mlruns"
print(f"Setting MLflow tracking URI to: {tracking_uri}") # Debugging
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("CI-CD-Lab")

# --- Ejecución de MLflow ---
with mlflow.start_run() as run:
    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"MLflow Artifact URI: {run.info.artifact_uri}") # Debugging
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse", mse)

    # Guarda el modelo en la raíz del proyecto (donde se ejecuta 'make')
    # Esto coincide con lo que espera el paso de upload-artifact del workflow
    model_path = "model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved locally to: {os.path.abspath(model_path)}") # Debugging

    # Registra el artefacto desde la raíz del proyecto
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Artifact logged to MLflow path: model") # Debugging

print(f"✅ Entrenamiento completo. MSE: {mse:.4f}")