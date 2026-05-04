import os
import mlflow
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature

print("Starting Advanced Pipeline Training...")

# 1. Generate Realistic Industrial Data
np.random.seed(42)
n_samples = 5000

data = pd.DataFrame({
    'Air_Temperature': np.random.normal(300, 2, n_samples),
    'Process_Temperature': np.random.normal(310, 1.5, n_samples),
    'Rotational_Speed': np.random.normal(1500, 100, n_samples),
    'Torque': np.random.normal(40, 5, n_samples),
    'Tool_Wear': np.random.uniform(0, 250, n_samples)
})

# Base Failure Conditions
failures = ((data['Tool_Wear'] > 200) & (data['Torque'] > 45)) | (data['Rotational_Speed'] < 1300)
data['Machine_Failure'] = failures.astype(int)

# --- THE FIX: Add Real-World Noise ---
# Flip 7% of the labels randomly to mimic real-world unpredictability
# This makes it IMPOSSIBLE for the model to get a 1.00 (100%) score!
noise_indices = np.random.choice(data.index, size=int(n_samples * 0.07), replace=False)
data.loc[noise_indices, 'Machine_Failure'] = 1 - data.loc[noise_indices, 'Machine_Failure']

X = data.drop('Machine_Failure', axis=1)
y = data['Machine_Failure']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Configure MLflow (Absolute Path Fix)
# Yeh line MLflow ko force karegi ke wo current folder mein hi mlflow.db banaye/use kare
db_path = os.path.join(os.getcwd(), "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment("Industrial_Predictive_Maintenance_V2")

print(f"Tracking URI set to: {mlflow.get_tracking_uri()}")

with mlflow.start_run() as run:
    print(f"Run ID: {run.info.run_id}")
    
    # Define XGBoost Parameters
    params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "scale_pos_weight": 8 
    }
    
    # 3. Create a Professional ML Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(**params, random_state=42))
    ])
    
    # 4. Train the Pipeline
    print("Training XGBoost Pipeline...")
    pipeline.fit(X_train, y_train)
    
    # 5. Evaluate the Model
    y_pred = pipeline.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # 6. Log to MLflow
    mlflow.log_params(params)
    mlflow.log_metrics({"f1_score": f1, "precision": precision, "recall": recall})
    
    # Create Signature 
    signature = infer_signature(X_train, y_pred)
    
    # Log the complete pipeline
    mlflow.sklearn.log_model(
        sk_model=pipeline, 
        artifact_path="maintenance_pipeline", 
        signature=signature,
        input_example=X_train.iloc[:5] 
    )

print("\nProfessional Training Complete!")