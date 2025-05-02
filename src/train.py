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

print(f"--- Debug: Initial CWD: {os.getcwd()} ---") # Debería ser la raíz del proyecto

# Este path está dentro del workspace de GitHub Actions
# Usar ruta absoluta basada en CWD para mayor claridad
mlflow_tracking_dir = os.path.abspath(os.path.join(os.getcwd(), "mlruns"))
print(f"--- Debug: Absolute tracking dir path: {mlflow_tracking_dir} ---")
os.makedirs(mlflow_tracking_dir, exist_ok=True) # Crear si no existe
tracking_uri = "file://" + mlflow_tracking_dir
print(f"--- Debug: Setting Tracking URI to: {tracking_uri} ---")
mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("CI-CD-Lab")

# Cargar datos y entrenar modelo
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)

# Signature and input example (comentados temporalmente para simplificar)
# signature = infer_signature(X_train, model.predict(X_train))
# input_example = pd.DataFrame(X_train).head(3)

print("--- Debug: Starting MLflow run ---")
run = None # Inicializar run fuera del try para el bloque except
try:
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri # ¡Esta es la ruta clave!
        print(f"--- Debug: Run ID: {run_id} ---")
        # Verificar que esta URI apunte a mlflow_tracking_dir
        print(f"--- Debug: Run Artifact URI: {artifact_uri} ---")
        if not artifact_uri.startswith(tracking_uri):
             print(f"--- WARNING: Artifact URI does not match Tracking URI! ---")


        mlflow.log_metric("mse", mse)
        print(f"--- Debug: Attempting log_model with artifact_path='model' (no signature/example) ---")

        # Llamada simplificada
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model" # Ruta relativa dentro de artifact_uri
            # signature=signature,      # Comentado
            # input_example=input_example # Comentado
        )
        print(f"✅ Modelo registrado correctamente (sin firma/ejemplo). MSE: {mse:.4f}")

except Exception as e:
    print(f"\n--- ERROR during MLflow execution ---")
    traceback.print_exc() # Imprimir traza completa
    print(f"--- End Error Traceback ---")
    # Imprimir contexto adicional
    print(f"Current CWD at error: {os.getcwd()}")
    print(f"Tracking URI used: {mlflow.get_tracking_uri()}")
    if run:
         print(f"Run Artifact URI at error: {run.info.artifact_uri}")
    else:
         print("Run object was not successfully created.")
    sys.exit(1) # Fallar explícitamente
