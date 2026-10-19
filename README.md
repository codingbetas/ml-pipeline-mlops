# End-to-End MLOps Pipeline

A production-grade Machine Learning Operations (MLOps) pipeline. This project demonstrates the complete lifecycle of a machine learning model: from training and experiment tracking to versioning and serving it as a live REST API.

## 🧠 System Design Concepts Demonstrated

1. **Experiment Tracking:** Uses MLflow to track hyperparameters (e.g., `n_estimators`), evaluation metrics (accuracy), and model artifacts.
2. **Model Registry:** Packages the trained model (RandomForest) and its dependencies into a portable artifact in the MLflow registry.
3. **Model Serving (Inference):** A FastAPI server dynamically loads the specific model version from the MLflow registry and exposes it via a `/predict` endpoint with Pydantic validation.
4. **Containerized Infrastructure:** Runs the MLflow tracking server in Docker for isolated, reproducible environments.

## 🏗️ Architecture

```text
       ┌──────────────┐
       │  app/train   │ (Trains RandomForest on Iris dataset)
       └──────┬───────┘
              │ 1. Logs metrics & saves model
              ▼
       ┌──────────────┐
       │    MLflow    │ (Dockerized Tracking Server)
       └──────┬───────┘
              │ 2. Loads specific RUN_ID
              ▼
       ┌──────────────┐
       │  app/main    │ (FastAPI Inference Server)
       └──────┬───────┘
              │ 3. POST /predict
              ▼
       ┌──────────────┐
       │    Client    │ (Receives predicted class)
       └──────────────┘
```

## 🛠️ Tech Stack

- **ML Framework:** `scikit-learn`
- **MLOps Tracking:** MLflow
- **API Framework:** FastAPI
- **Orchestration:** Docker Compose
- **Data Processing:** pandas

## 🚀 How to Run

**1. Start the MLflow Tracking Server:**
```bash
docker compose up -d
```
*View the MLflow UI at `http://localhost:5000`*

**2. Install dependencies:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**3. Train and log the model:**
```bash
python -m app.train
```
*Look at the terminal output. It will give you a `RUN_ID` (e.g., `da75fa9135cb...`).*

**4. Serve the model:**
Copy the `RUN_ID` from step 3 and paste it into the `RUN_ID` variable in `app/main.py`. Then run:
```bash
uvicorn app.main:app --reload
```

**5. Test the API:**
Go to `http://localhost:8000/docs` and test the `/predict` endpoint with:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

## 📄 License

MIT