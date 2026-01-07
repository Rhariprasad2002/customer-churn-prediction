from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load trained artifacts
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "Customer Churn Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400

        input_values = []

        # Maintain training feature order
        for col in feature_columns:
            if col not in data:
                return jsonify({"error": f"Missing feature: {col}"}), 400

            value = data[col]

            # Encode categorical values
            if col in label_encoders:
                try:
                    value = label_encoders[col].transform([value])[0]
                except ValueError:
                    return jsonify({"error": f"Invalid value for {col}: {value}"}), 400

            input_values.append(value)

        # Convert to NumPy array
        input_array = np.array(input_values, dtype=float).reshape(1, -1)

        # Scale features
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = model.predict(input_scaled)[0]

        return jsonify({
            "prediction": int(prediction),
            "result": "Churn" if prediction == 1 else "No Churn"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # IMPORTANT: debug=False to avoid Windows reload crash
    app.run(host="127.0.0.1", port=5000, debug=False)
