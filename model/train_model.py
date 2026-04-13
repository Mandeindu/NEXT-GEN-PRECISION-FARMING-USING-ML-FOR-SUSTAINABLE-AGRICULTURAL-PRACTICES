import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import joblib
import os

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv('dataset/precision_farming_data.csv')

# 2. Preprocessing
print("Preprocessing data...")
# Categorical columns to encode
cat_cols = ['Soil Type', 'Season', 'Location', 'Crop']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders for inference
joblib.dump(encoders, 'model/encoders.pkl')

# Define features and target
X = df.drop('Crop', axis=1).values
y = df['Crop'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
print("Training TabNet model...")
# Define TabNet Classifier
clf = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax' # "sparsemax"
)

# Fit the model
clf.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_name=['train', 'valid'],
    eval_metric=['accuracy'],
    max_epochs=100 , patience=20,
    batch_size=64, virtual_batch_size=8,
    num_workers=0,
    drop_last=False
)

# 4. Evaluation
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, average='weighted', zero_division=0)
rec = recall_score(y_test, preds, average='weighted', zero_division=0)
f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

print(f"\nModel Performance:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save metrics
with open('model/metrics.txt', 'w') as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall: {rec:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

# 5. Save Model
save_path = 'model/tabnet_model'
clf.save_model(save_path)
print(f"Model saved to {save_path}.zip")
