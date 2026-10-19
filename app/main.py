import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="MLOps Inference API")

# Connect to MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Load the latest model from the registry
# We use the 'runs:/' URI to load the specific run we just did
RUN_ID = "da75fa9135cb4e878901f690d8eef764" # <--- REPLACE THIS WITH YOUR RUN ID
model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/random_forest_model")

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def health():
    return {"status": "ok", "service": "MLOps Inference API"}

@app.post("/predict")
def predict(features: IrisFeatures):
    # Convert input to a DataFrame
    data = pd.DataFrame([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]], columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

    # Make prediction
    prediction = model.predict(data)[0]
    
    # Map the prediction to the flower name
    classes = ["Setosa", "Versicolor", "Virginica"]
    
    return {
        "prediction": int(prediction),
        "flower": classes[int(prediction)]
    }