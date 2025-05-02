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
import os

# Asegura que todo lo que se guarde, se quede dentro del repo
os.chdir(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri("file://" + os.path.abspath("../mlruns"))

mlflow.set_experiment("CI-CD-Lab")




with mlflow.start_run():
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse", mse)
    model_path = "model.pkl"
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact(model_path, artifact_path="model") # Guarda model.pkl dentro de una carpeta 'model' en los artefactos

print(f"✅ Entrenamiento completo. MSE: {mse:.4f}")
