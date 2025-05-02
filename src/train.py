import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from mlflow.models import infer_signature
import sys
import traceback

print(f"--- Debug: Initial CWD: {os.getcwd()} ---")

# --- Define Paths ---
# Usar rutas absolutas dentro del workspace del runner
workspace_dir = os.getcwd() # Debería ser /home/runner/work/mlflow-deploy/mlflow-deploy
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)
# Definir explícitamente la ubicación base deseada para los artefactos
artifact_location = "file://" + os.path.abspath(mlruns_dir)

print(f"--- Debug: Workspace Dir: {workspace_dir} ---")
print(f"--- Debug: MLRuns Dir: {mlruns_dir} ---")
print(f"--- Debug: Tracking URI: {tracking_uri} ---")
print(f"--- Debug: Desired Artifact Location Base: {artifact_location} ---")

# --- Asegurar que el directorio MLRuns exista ---
os.makedirs(mlruns_dir, exist_ok=True)

# --- Configurar MLflow ---
mlflow.set_tracking_uri(tracking_uri)

# --- Crear o Establecer Experimento Explícitamente con Artifact Location ---
experiment_name = "CI-CD-Lab2"
try:
    # Intentar crear el experimento, proporcionando la ubicación del artefacto
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location # ¡Forzar la ubicación aquí!
    )
    print(f"--- Debug: Creado Experimento '{experiment_name}' con ID: {experiment_id} ---")
except mlflow.exceptions.MlflowException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"--- Debug: Experimento '{experiment_name}' ya existe. Estableciéndolo. ---")
        mlflow.set_experiment(experiment_name=experiment_name)
        # Verificar la ubicación del artefacto del experimento existente
        experiment = mlflow.get_experiment_by_name(experiment_name)
        print(f"--- Debug: Ubicación de Artefacto del Experimento Existente: {experiment.artifact_location} ---")
        # Si la ubicación existente es incorrecta, esta es probablemente la causa raíz
        if experiment.artifact_location != artifact_location:
             print(f"--- ¡¡¡ADVERTENCIA CRÍTICA!!!: La ubicación del artefacto del experimento existente ('{experiment.artifact_location}') NO coincide con la deseada ('{artifact_location}')! Esto probablemente causa el error. ---")
             # Considera eliminar el directorio mlruns o el experimento si esto ocurre persistentemente.
    else:
        print(f"--- ERROR creando/estableciendo experimento: {e} ---")
        raise e # Relanzar otros errores

# --- Cargar Datos y Entrenar Modelo ---
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

# --- Iniciar Run de MLflow ---
print("--- Debug: Iniciando run de MLflow ---")
run = None
try:
    # Iniciar el run dentro del experimento configurado
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        # Verificar la URI del artefacto OTRA VEZ ahora que el run ha iniciado
        actual_artifact_uri = run.info.artifact_uri
        print(f"--- Debug: Run ID: {run_id} ---")
        print(f"--- Debug: URI Real del Artefacto del Run: {actual_artifact_uri} ---")

        # Comprobar si coincide con el patrón esperado basado en tracking_uri
        if not actual_artifact_uri.startswith(tracking_uri):
             print(f"--- ¡¡¡ADVERTENCIA CRÍTICA!!!: La URI del Artefacto del Run '{actual_artifact_uri}' TODAVÍA no coincide con la Tracking URI '{tracking_uri}'! ---")
             # Esto indica un problema más profundo de MLflow o una configuración incorrecta persistente.

        mlflow.log_metric("mse", mse)
        print(f"--- Debug: Intentando log_model con artifact_path='model' ---")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )
        print(f"✅ Modelo registrado correctamente. MSE: {mse:.4f}")

except Exception as e:
    print(f"\n--- ERROR durante la ejecución de MLflow ---")
    traceback.print_exc()
    print(f"--- Fin de la Traza de Error ---")
    print(f"CWD actual en el error: {os.getcwd()}")
    print(f"Tracking URI usada: {mlflow.get_tracking_uri()}")
    if run:
         print(f"URI del Artefacto del Run en el error: {run.info.artifact_uri}") # Imprimir de nuevo
    else:
         print("El objeto Run no se creó con éxito.")
    sys.exit(1)
