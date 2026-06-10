import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
import joblib
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
df = pd.read_csv(r"C:\Users\shiva\Downloads\combined_cleaned.csv")
X = df.drop('Attrition', axis=1)
y = df['Attrition']
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
if y.dtype == 'object':
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
imputer = SimpleImputer(strategy="most_frequent")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
mlp_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=500,
    random_state=42
)
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss"
)
stacked_model = StackingClassifier(
    estimators=[
        ('mlp', mlp_model),
        ('xgb', xgb_model)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)
stacked_model.fit(X_train, y_train)
print("✅ Stacked model trained successfully!")
cv_scores = cross_val_score(stacked_model, X, y, cv=5, scoring='accuracy')
cv_accuracy_mean = cv_scores.mean()
cv_accuracy_std = cv_scores.std()
save_dir = r"C:\Users\shiva\Desktop\dsa_pro"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
stacked_model_path = os.path.join(save_dir, "stacked_model.pkl")
joblib.dump(stacked_model, stacked_model_path, compress=3)
print(f"✅ Stacked model saved at: {stacked_model_path}")
y_pred = stacked_model.predict(X_test)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred
print("\n📊 Stacked Model Performance:\n")
print(f"Training Accuracy: {stacked_model.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {stacked_model.score(X_test, y_test):.4f}")
print(f"CV Accuracy Mean: {cv_accuracy_mean:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stay','Leave'], yticklabels=['Stay','Leave'])
plt.title('Stacked Model Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
metrics = [
    stacked_model.score(X_train, y_train),
    stacked_model.score(X_test, y_test),
    cv_accuracy_mean,
    precision,
    recall,
    f1
]
metric_names = ['Train Acc', 'Test Acc', 'CV Acc', 'Precision', 'Recall', 'F1-Score']
plt.figure(figsize=(8,4))
plt.bar(metric_names, metrics, color=['skyblue','blue','purple','orange','green','red'])
plt.ylim(0,1)
plt.title('Stacked Model Performance Metrics')
plt.ylabel('Score')
plt.show()
