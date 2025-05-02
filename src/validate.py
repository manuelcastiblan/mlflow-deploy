# src/validate.py
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sys

# Parámetro de umbral
THRESHOLD = 20.0  # cambia este valor según tu dataset y expectativas

# Simulamos el mismo dataset
df = pd.DataFrame({
    "x": range(100),
    "y": [2*i + 3 + (i % 5) for i in range(100)]
})

X = df[["x"]]
y = df["y"]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Cargar modelo previamente entrenado
model = joblib.load("model.pkl")
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
