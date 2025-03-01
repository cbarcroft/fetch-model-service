FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt 

COPY docker/download_model.py .
RUN python /app/download_model.py

COPY . .

CMD ["python", "main.py"]