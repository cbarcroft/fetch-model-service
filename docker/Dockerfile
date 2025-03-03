FROM python:3.11-slim

WORKDIR /app

# Install dependencies used in model download steps.
RUN pip install --no-cache-dir transformers torch "numpy<2" \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY config.ini .

# Download transformer model from Huggingface
COPY docker/download_hf_model.py .
RUN python /app/download_hf_model.py

# Download ONNX version of same model
COPY docker/download_onnx_model.py .
RUN python /app/download_onnx_model.py

# Install app dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt 

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]