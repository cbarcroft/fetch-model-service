from fastapi import FastAPI, HTTPException
from fastapi_cache.decorator import cache
import requests
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer, pipeline
import os
import configparser

from app.models.inference_request_model import InferenceRequestModel
from app.utils.softmax import softmax

config = configparser.ConfigParser()
config.read("config.ini")

app = FastAPI()

# Load transformer model from Huggingface
sentiment_pipeline = pipeline("sentiment-analysis", model=config['transformer']['local_path'], tokenizer=config['transformer']['local_path'])

# Print sentiment of the success message.  Should be positive :)
print(sentiment_pipeline("Transformer model initialized successfully."))

# Load Tokenizer (Same as original model)
TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load ONNX model with ONNX Runtime
print("Loading ONNX model...")
session = ort.InferenceSession(config['onnx']['filename'])
print("ONNX model initialized successfully.")

# Healthcheck endpoint for kubernetes; simply responds 200 with status OK
@app.get("/health")
async def health_check():
    return {"status": "OK"}

# Inferance endpoint for HF model in pipeline.
@app.post("/transformers/infer")
@cache(expire=60)
def infer(request: InferenceRequestModel):
	sentiment_response = sentiment_pipeline(request.input)

	return sentiment_response

# Inference endpoint for ONNX version of same model.  Performs better, but requires 
# a little more groundwork in both creating the input and parsing the result.
@app.post("/onnx/infer")
@cache(expire=60)
async def infer(request: InferenceRequestModel):
    try:
        inputs = TOKENIZER(request.input, return_tensors="np")
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)

        # Run inference
        outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

        # Extract logits and convert to probability to match transformer model
        logits = outputs[0][0]
        probabilities = softmax(logits)

        # Make prediction
        labels = ["NEGATIVE", "POSITIVE", "NEUTRAL"]
        predicted_class = np.argmax(probabilities)
        sentiment = labels[predicted_class]
        confidence_score = round(float(probabilities[predicted_class]), 15)  # Format score to match transformer model

        return {"sentiment": sentiment, "score": confidence_score}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
