import os
import requests

MODEL_URL = "https://huggingface.co/onnx-community/distilbert-base-uncased-finetuned-sst-2-english-ONNX/resolve/main/onnx/model.onnx"
MODEL_PATH = "model.onnx"

if not os.path.exists(MODEL_PATH):
    print("Downloading ONNX model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print(f"Downloaded ONNX model to {MODEL_PATH}")