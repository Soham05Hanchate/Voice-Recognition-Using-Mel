"""Local web app for the voice-command models trained in the notebook."""

from pathlib import Path
import os

import librosa
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file


ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("VOICE_MODEL_DIR", ROOT / "models"))
COMMANDS = ["up", "down", "left", "right", "go", "stop"]
SR = 16000
CLIP_SAMPLES = SR
N_MELS, HOP, N_FFT = 64, 256, 512
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
_models = {}


def extract_features(y):
    """Keep this feature definition aligned with notebook cell 11."""
    features = [
        librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13),
        librosa.feature.zero_crossing_rate(y=y),
        librosa.feature.spectral_centroid(y=y, sr=SR),
        librosa.feature.spectral_bandwidth(y=y, sr=SR),
        librosa.feature.spectral_rolloff(y=y, sr=SR),
    ]
    output = []
    for feature in features:
        output.extend(np.mean(feature, axis=1))
        output.extend(np.std(feature, axis=1))
    return np.asarray(output, dtype=np.float32)


def extract_logmel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=100.0).astype(np.float32)


def load_models():
    """Load available artifacts once; CNNs are optional for SVM-only installs."""
    if _models:
        return _models
    try:
        import joblib
        for filename, label in [
            ("noise_augmented_svm.joblib", "Noise-augmented SVM"),
            ("clean_trained_svm.joblib", "Clean-trained SVM"),
        ]:
            path = MODEL_DIR / filename
            if path.is_file():
                _models[label] = ("svm", joblib.load(path))
    except Exception as exc:
        app.logger.exception("Could not load SVM model: %s", exc)

    # CNN loading is independent so a TensorFlow issue does not disable SVMs.
    try:
        import tensorflow as tf
        for filename, label in [
            ("noise_augmented_cnn.keras", "Noise-augmented CNN"),
            ("clean_trained_cnn.keras", "Clean-trained CNN"),
        ]:
            path = MODEL_DIR / filename
            if path.is_file():
                _models[label] = ("cnn", tf.keras.models.load_model(path))
    except Exception as exc:
        app.logger.warning("CNN models unavailable (TensorFlow load failed): %s", exc)
    return _models


def predict_audio(audio_bytes, model_name=None):
    import soundfile as sf
    audio, source_sr = sf.read(audio_bytes, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if audio.size == 0:
        raise ValueError("The audio recording is empty.")
    if source_sr != SR:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=SR)
    audio = librosa.util.fix_length(audio, size=CLIP_SAMPLES).astype(np.float32)

    available = load_models()
    if not available:
        raise RuntimeError(
            f"No model files found in {MODEL_DIR}. Copy the exported .joblib or .keras files there."
        )
    if model_name not in available:
        model_name = next(iter(available))
    kind, model = available[model_name]
    if kind == "svm":
        vector = extract_features(audio).reshape(1, -1)
        label = str(model.predict(vector)[0])
        probabilities = model.predict_proba(vector)[0]
        classes = model.named_steps["svc"].classes_
        scores = {str(k): float(v) for k, v in zip(classes, probabilities)}
    else:
        vector = extract_logmel(audio)[np.newaxis, ..., np.newaxis]
        probabilities = model.predict(vector, verbose=0)[0]
        label = COMMANDS[int(np.argmax(probabilities))]
        scores = {command: float(score) for command, score in zip(COMMANDS, probabilities)}
    return {"command": label, "model": model_name, "scores": scores}


@app.get("/")
def index():
    models = load_models()
    results_file = MODEL_DIR / "accuracy_by_noise_level_svm_vs_cnn.csv"
    return render_template("index.html", models=list(models), has_results=results_file.is_file())


@app.get("/results.csv")
def results_csv():
    path = MODEL_DIR / "accuracy_by_noise_level_svm_vs_cnn.csv"
    if not path.is_file():
        return jsonify(error="The model accuracy CSV is not installed."), 404
    return send_file(path, as_attachment=True, download_name=path.name, mimetype="text/csv")


@app.post("/api/predict")
def api_predict():
    file = request.files.get("audio")
    if file is None or not file.filename:
        return jsonify(error="Choose or record an audio clip first."), 400
    try:
        result = predict_audio(file.stream, request.form.get("model"))
        return jsonify(result)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 503
    except (ValueError, librosa.util.exceptions.ParameterError) as exc:
        return jsonify(error=f"Could not read this audio: {exc}"), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction failed. Check the server log for details."), 500


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(error="Audio file is too large (maximum 10 MB)."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
