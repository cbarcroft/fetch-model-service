import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline


print("Initialize tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
print("Initialize model...")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

print("Tokenize input text...")
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

print("Predict...")
with torch.no_grad():
    logits = model(**inputs).logits

print("Get results...")
predicted_class_id = logits.argmax().item()
res = model.config.id2label[predicted_class_id]

print(res)
# Code only change