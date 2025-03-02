from fastapi import FastAPI
from transformers import pipeline

from app.models.inference_request_model import InferenceRequestModel

sentiment_pipeline = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

print(sentiment_pipeline("Model initialized successfully."))

app = FastAPI()

@app.post("/infer")
def infer(request: InferenceRequestModel):
	sentiment_response = sentiment_pipeline(request.input)

	return sentiment_response

@app.get("/health")
def health():
	return { 
		"status": "UP" 
	}