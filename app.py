
import os
import logging
import joblib
import pandas as pd

from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("student-result-api")

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "student_result_model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
    logger.info("event=model_loaded model_file=%s", MODEL_PATH)
except Exception:
    model = None
    logger.exception("event=model_load_failed")

# Automatically collect Flask request metrics.
metrics = PrometheusMetrics(app, path=None)


@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    }), 200


@app.route("/health")
def health():
    if model is None:
        return jsonify({
            "status": "unhealthy",
            "message": "Model file could not be loaded"
        }), 503

    return jsonify({"status": "healthy"}), 200


@app.route("/metrics")
def prometheus_metrics():
    try:
        return generate_latest(), 200, {
            "Content-Type": CONTENT_TYPE_LATEST
        }
    except Exception:
        logger.exception("event=metrics_generation_failed")
        return jsonify({"error": "Metrics generation failed"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Provide a valid JSON object"}), 400

    features = [
        "attendance",
        "internal_marks",
        "assignment_marks",
        "previous_score"
    ]

    missing = [key for key in features if key not in data]
    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    try:
        values = {key: float(data[key]) for key in features}
        input_df = pd.DataFrame([values], columns=features)
        prediction = int(model.predict(input_df)[0])

        result = "PASS" if prediction == 1 else "FAIL"
        logger.info("event=prediction_success result=%s", result)

        return jsonify({"prediction": result}), 200

    except (ValueError, TypeError):
        logger.warning("event=validation_error reason=non_numeric_features")
        return jsonify({
            "error": "All input features must be numeric"
        }), 400

    except Exception:
        logger.exception("event=prediction_failed")
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    logger.info("event=application_starting port=5000")
    app.run(host="0.0.0.0", port=5000)
