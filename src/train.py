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


# --- Configuración de MLflow ---
tracking_uri = ""
if os.getenv("GITHUB_ACTIONS") == "true":
    # Usar GITHUB_WORKSPACE para la ruta absoluta en Actions
    github_workspace = os.environ.get("GITHUB_WORKSPACE")
    if github_workspace:
        # Construir la ruta absoluta para mlruns dentro del workspace
        mlruns_path = os.path.join(github_workspace, "mlruns")
        # Crear la URI de seguimiento (formato file:// + ruta absoluta)
        tracking_uri = "file://" + os.path.abspath(mlruns_path)
        print(f"GitHub Actions: Usando tracking URI: {tracking_uri}")
    else:
        # Fallback si GITHUB_WORKSPACE no está definido (poco probable)
        tracking_uri = "file://" + os.path.abspath("mlruns")
        print(f"GitHub Actions (Warning: GITHUB_WORKSPACE no encontrado): Usando tracking URI: {tracking_uri}")
else:
    # Ejecución local: Usar una ruta relativa es generalmente preferible
    # Asume que el script se ejecuta desde la raíz del proyecto o el Makefile gestiona el CWD
    tracking_uri = "file://" + os.path.abspath("mlruns")
    # O si necesitas la ruta absoluta local específica:
    # tracking_uri = "file:///home/manuelcastiblan/academic/mlflow-deploy/mlflow-deploy/mlruns"
    print(f"Local: Usando tracking URI: {tracking_uri}")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("CI-CD-Lab")




with mlflow.start_run():
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse", mse)
    joblib.dump(model, "model.pkl")
    mlflow.sklearn.log_model(model, "model", registered_model_name="ci-cd-model")

print(f"✅ Entrenamiento completo. MSE: {mse:.4f}")
