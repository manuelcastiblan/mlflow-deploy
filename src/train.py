import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from mlflow.models import infer_signature


# Este path está dentro del workspace de GitHub Actions
mlflow_tracking_dir = os.path.join(os.getcwd(), "mlruns")
os.makedirs(mlflow_tracking_dir, exist_ok=True)
mlflow.set_tracking_uri("file://" + mlflow_tracking_dir)



mlflow.set_experiment("CI-CD-Lab")

# Cargar datos y entrenar modelo
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)

# ✅ Ejemplo para inferir firma y loggear desde memoria
signature = infer_signature(X_train, model.predict(X_train))
input_example = pd.DataFrame(X_train).head(3)

with mlflow.start_run():
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )
    print(f"✅ Modelo registrado correctamente. MSE: {mse:.4f}")
