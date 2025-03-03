from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import requests
from huggingface_hub import configure_http_backend

import configparser

config = configparser.ConfigParser()
config.read("config.ini")

print(config)


def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

def download_hf_model(model_path, model_name):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


download_hf_model(config["transformer"]["local_path"], config["transformer"]["model_name"])
