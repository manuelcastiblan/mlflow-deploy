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

# Construir una ruta absoluta para mlruns dentro del workspace de GitHub Actions
workspace_path = os.environ.get("GITHUB_WORKSPACE", ".") # Usar "." como fallback si no está en Actions
mlruns_path = os.path.join(workspace_path, "mlruns")
# Crear la URI de seguimiento con la ruta absoluta (asegurándose de que sea un URI de archivo válido)
# No es necesario 'file:///' si os.path.join ya da una ruta absoluta POSIX
tracking_uri = "file:" + os.path.abspath(mlruns_path)

print(f"Setting MLflow tracking URI to: {tracking_uri}") # Añadir log para depuración
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("CI-CD-Lab")



with mlflow.start_run():
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse", mse)
    joblib.dump(model, "model.pkl")
    mlflow.sklearn.log_model(model, "model", registered_model_name="ci-cd-model")

print(f"✅ Entrenamiento completo. MSE: {mse:.4f}")
