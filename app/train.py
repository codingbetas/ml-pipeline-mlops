import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Connect to the MLflow server running in Docker
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Iris_Classification")

def train_model():
    print("🚀 Loading data...")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    print("🧠 Training model...")
    with mlflow.start_run():
        # Train the model
        model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        # Log parameters and metrics to MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 3)
        mlflow.log_metric("accuracy", acc)
        
        # Save the model
        mlflow.sklearn.log_model(
           model, 
           "random_forest_model", 
           skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        
        print(f"✅ Model trained with accuracy: {acc:.4f}")
        print("📦 Model saved to MLflow Registry.")

if __name__ == "__main__":
    train_model()