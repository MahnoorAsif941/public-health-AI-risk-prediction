"""
Title: Epidemiological Risk Prediction using Random Forest
Author: Mahnoor Asif
Description: Simple, interpretable machine learning model for public health risk assessment.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Simulate Public Health Cohort Data (e.g., n=1000)
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'age': np.random.randint(30, 80, size=n_samples),
    'bmi': np.random.uniform(18.5, 35.0, size=n_samples),
    'daily_pm25_exposure': np.random.uniform(5, 90, size=n_samples), # Environmental exposure
    'smoking_status': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
    'hypertension': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
})

# Generate target outcome (cardiovascular/respiratory risk event)
latent_risk = (
    0.05 * data['age'] + 
    0.08 * data['bmi'] + 
    0.02 * data['daily_pm25_exposure'] + 
    1.5 * data['smoking_status'] + 
    1.2 * data['hypertension'] - 7.0
)
probability = 1 / (1 + np.exp(-latent_risk))
data['risk_event'] = np.random.binomial(1, probability)

# 2. Define Features (X) and Target (y)
X = data[['age', 'bmi', 'daily_pm25_exposure', 'smoking_status', 'hypertension']]
y = data['risk_event']

# 3. Split into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Model
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("--- Model Performance Report ---")
print(classification_report(y_test, predictions))
print(f"ROC-AUC Score: {roc_auc_score(y_test, probabilities):.3f}")

# 6. Feature Importance (Interpretability is critical in public health)
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- Feature Importances ---")
print(importance)
