FROM python:3.11-slim

WORKDIR /app

COPY requirements_hf.txt .
RUN pip install -r requirements_hf.txt 

COPY docker/download_model.py .
RUN python /app/download_model.py

COPY requirements.txt .
RUN pip install -r requirements.txt 

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]