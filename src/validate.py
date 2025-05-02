# src/validate.py
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sys
import os # Importar os

# Parámetro de umbral
THRESHOLD = 20.0  # cambia este valor según tu dataset y expectativas

# Simulamos el mismo dataset
# ... (código del dataset y split como antes) ...
df = pd.DataFrame({
    "x": range(100),
    "y": [2*i + 3 + (i % 5) for i in range(100)]
})

X = df[["x"]]
y = df["y"]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# --- Cargar modelo previamente entrenado ---
# Construir la ruta absoluta al modelo esperado en la raíz del proyecto
model_filename = "model.pkl"
# os.getcwd() debería ser la raíz del proyecto cuando se ejecuta con 'make'
model_path = os.path.abspath(os.path.join(os.getcwd(), model_filename))

print(f"--- Debug: Intentando cargar modelo desde: {model_path} ---") # Añadir debug

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"--- ERROR: No se encontró el archivo del modelo en '{model_path}'. Asegúrate de que el paso 'make train' lo haya guardado correctamente en la raíz del proyecto. ---")
    # Listar archivos en el directorio actual para depuración
    print(f"--- Debug: Archivos en {os.getcwd()}: ---")
    try:
        print(os.listdir(os.getcwd()))
    except Exception as list_err:
        print(f"(No se pudo listar el directorio: {list_err})")
    print("---")
    sys.exit(1) # Salir con error

# --- Predicción y Validación ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"🔍 MSE del modelo: {mse:.4f} (umbral: {THRESHOLD})")

# Validación
if mse <= THRESHOLD:
    print("✅ El modelo cumple los criterios de calidad.")
    sys.exit(0)  # éxito
else:
    print("❌ El modelo no cumple el umbral. Deteniendo pipeline.")
    sys.exit(1)  # error