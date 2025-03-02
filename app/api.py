from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

sentiment_pipeline = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

print(sentiment_pipeline("Model initialized successfully."))

app = FastAPI()

class InferenceRequestModel(BaseModel):
   input: str

@app.post("/infer")
def infer(request: InferenceRequestModel):
	sentiment_response = sentiment_pipeline(request.input)

	return sentiment_response

@app.get("/health")
def health():
	return { status: "UP" }