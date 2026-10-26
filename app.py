
from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

MODEL_PATH = "student_result_model.pkl"

model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Please provide a valid JSON object"}), 400

    required_features = [
        "attendance",
        "internal_marks",
        "assignment_marks",
        "previous_score"
    ]

    missing = [feature for feature in required_features if feature not in data]

    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    try:
        values = [[float(data[feature]) for feature in required_features]]
        prediction = model.predict(values)[0]

        if hasattr(prediction, "item"):
            prediction = prediction.item()

        return jsonify({
            "prediction": str(prediction)
        }), 200

    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
