import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Configurar ruta de tracking (relativa, segura en GitHub Actions)
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.set_experiment("CI-CD-Lab")

# Cargar y dividir dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

# Iniciar experimento
with mlflow.start_run():
    mlflow.log_metric("mse", mse)

    # Loggear modelo directamente desde memoria
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="modelo-diabetes-ci"
    )

    print(f"✅ Modelo registrado con MSE: {mse:.4f}")
