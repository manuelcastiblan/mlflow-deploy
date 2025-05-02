import os
import mlflow
import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Tracking local (válido tanto en local como en Actions)
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.set_experiment("CI-CD-Lab")

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

model_path = "model.pkl"  # guardado en ruta local
joblib.dump(model, model_path)

with mlflow.start_run():
    mlflow.log_metric("mse", mse)
    mlflow.log_artifact(model_path, artifact_path="model")  # ✅ artefacto local, no apunta a ninguna ruta absoluta
    print(f"✅ Modelo entrenado. MSE: {mse}")
