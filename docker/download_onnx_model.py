import os
import requests

import configparser

config = configparser.ConfigParser()
config.read("config.ini")

MODE_FILENAME = config["onnx"]["filename"]


if not os.path.exists(MODE_FILENAME):
    print("Downloading ONNX model...")
    response = requests.get(config["onnx"]["source_url"])
    with open(MODE_FILENAME, "wb") as f:
        f.write(response.content)
    print(f"Downloaded ONNX model to {MODE_FILENAME}")