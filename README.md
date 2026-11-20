# Voice Command Lab

A local web app for testing the models exported by `Voice_Command_Recognition_Noise_SVM_vs_CNN.ipynb`. It supports microphone recordings and WAV uploads, displays the predicted command, and shows the model's class scores.

The app expects these notebook export filenames under `models/` (you can copy only the models you want to use):

| Model | Artifact |
| --- | --- |
| Clean-trained SVM | `clean_trained_svm.joblib` |
| Noise-augmented SVM | `noise_augmented_svm.joblib` |
| Clean-trained CNN | `clean_trained_cnn.keras` |
| Noise-augmented CNN | `noise_augmented_cnn.keras` |

The SVM models are loaded when present. CNN models are also loaded if TensorFlow is installed. Install TensorFlow separately to enable CNN support; the app remains usable with only SVM artifacts. Use the same compatible Python, scikit-learn, and TensorFlow versions used to export the artifacts, especially for serialized SVM pipelines.

## Run locally

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p models
# Copy your exported model files into models/
python app.py
```

Open <http://127.0.0.1:5000>. Microphone access is available on localhost in modern browsers. The model receives mono audio resampled to 16 kHz and fixed to the first second, matching notebook preprocessing. Supported commands are `up`, `down`, `left`, `right`, `go`, and `stop`.

To store models in another directory, set `VOICE_MODEL_DIR` before starting the app.

## Run with Docker

Docker builds the app on Python 3.11, includes all model artifacts in `models/`, and installs the TensorFlow version used to train the CNN (2.21.0). From the project directory:

```sh
docker compose up --build
```

Open <http://127.0.0.1:5000>. Stop it with `Ctrl+C`; run `docker compose down` to remove the container. The first build downloads Python packages and creates a larger image because TensorFlow is included. The container runs the app on CPU by default.
