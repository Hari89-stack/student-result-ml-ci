
from flask import Flask, request, jsonify, g
import joblib
import os
import logging
import sys
import time
import uuid

app = Flask(__name__)

# Configure logging to the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")

MODEL_PATH = "student_result_model.pkl"
model = None

# Load the trained model and log the result
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info("event=model_loaded model_file=%s", MODEL_PATH)
    else:
        logger.error("event=model_load_failed reason=model_file_missing")
except Exception:
    logger.exception("event=model_load_failed")
    raise


# Log each HTTP request and measure its duration
@app.before_request
def before_request():
    g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    g.start_time = time.perf_counter()


@app.after_request
def after_request(response):
    request_id = getattr(g, "request_id", str(uuid.uuid4()))
    duration_ms = (time.perf_counter() - getattr(
        g, "start_time", time.perf_counter()
    )) * 1000

    response.headers["X-Request-ID"] = request_id

    level = logging.WARNING if response.status_code >= 400 else logging.INFO

    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms
    )

    return response


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    logger.exception(
        "event=unexpected_error request_id=%s path=%s",
        getattr(g, "request_id", "unknown"),
        request.path
    )
    return jsonify({"error": "Internal server error"}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    }), 200


@app.route("/health", methods=["GET"])
def health():
    if model is None:
        logger.error("event=model_not_ready")
        return jsonify({
            "status": "unhealthy",
            "message": "Model file not found"
        }), 503

    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        logger.error("event=prediction_failed reason=model_not_loaded")
        return jsonify({"error": "Model is not loaded"}), 500

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        logger.warning("event=validation_error reason=invalid_json_object")
        return jsonify({"error": "Provide a valid JSON object"}), 400

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
        logger.warning(
            "event=validation_error missing_fields=%s",
            missing
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    try:
        values = [[float(data[feature]) for feature in required_features]]

        logger.info(
            "event=prediction_request feature_names=%s",
            required_features
        )

        prediction = model.predict(values)[0]

        if hasattr(prediction, "item"):
            prediction = prediction.item()

        if prediction == 1:
            result = "PASS"
        elif prediction == 0:
            result = "FAIL"
        else:
            result = str(prediction)

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
