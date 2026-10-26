
from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "student_result_model.pkl"
model = None

# Load the trained ML model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    }), 200


@app.route("/health", methods=["GET"])
def health():
    if model is None:
        return jsonify({
            "status": "unhealthy",
            "message": "Model file not found"
        }), 503

    return jsonify({
        "status": "healthy"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Model is not loaded"
        }), 500

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Provide a valid JSON object"
        }), 400

    required_features = [
        "attendance",
        "internal_marks",
        "assignment_marks",
        "previous_score"
    ]

    missing = [
        feature for feature in required_features
        if feature not in data
    ]

    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    try:
        values = [[
            float(data[feature])
            for feature in required_features
        ]]

        # Generate prediction
        prediction = model.predict(values)[0]

        if hasattr(prediction, "item"):
            prediction = prediction.item()

        # Convert binary predictions into result labels
        if prediction == 1:
            result = "PASS"
        elif prediction == 0:
            result = "FAIL"
        else:
            result = str(prediction)

        return jsonify({
            "prediction": result
        }), 200

    except (ValueError, TypeError):
        return jsonify({
            "error": "All input features must be numeric"
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed"
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
