from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

# -------- Load model safely --------
current_folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_folder, "saved_models", "stacked_boosting_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = joblib.load(model_path)
print(f"✅ Model loaded from: {model_path}")

# -------- Load dataset to get feature names --------
csv_path = os.path.join(current_folder, "employee_attrition_dataset_cleaned.csv")
df = pd.read_csv(csv_path)
feature_names = df.drop("Attrition", axis=1).columns.tolist()

@app.route('/')
def home():
    return render_template("index.html", feature_names=feature_names, prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect values in the same order as feature_names
        data = [float(request.form[feature]) for feature in feature_names]
        final_input = np.array(data).reshape(1, -1)

        prediction = model.predict(final_input)[0]
        return render_template("index.html", feature_names=feature_names,
                               prediction_text=f"Prediction: {prediction}")
    except Exception as e:
        return render_template("index.html", feature_names=feature_names,
                               prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)