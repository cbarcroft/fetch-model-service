# Sentiment Analysis Service

Streamlit URL: http://k8s-default-streamli-315ff77678-d0c2616eb67d268c.elb.us-west-2.amazonaws.com/

API URL: http://k8s-default-fetchmod-efee52b36f-cf58d7cb3e601b77.elb.us-west-2.amazonaws.com/

## Overview
This project is a simple cloud-based Sentiment Analysis Service designed as a quick demonstration of cloud-based model deployment. It consists of a **FastAPI backend** for processing requests, a **Streamlit-based frontend** for interactive model comparison, and a **CI/CD pipeline** using GitHub Actions for automated deployment to AWS. The service is containerized with Docker and deployed on **Amazon EKS (Elastic Kubernetes Service)** for scalability and reliability.

Running the same query through two versions of the same model displays the performance gains provided by ONNX.  You will generally see inference response times 50%+ faster than their non-ONNX counterpart.

## Components
- **FastAPI Backend**: Provides an API endpoint for sentiment analysis using two in-memory versions of one model, one torch-based transformer version and one ONNX version.
- **Streamlit Frontend**: Allows users to compare sentiment predictions and performance from both model versions interactively.
- **AWS Deployment**: The service is containerized with Docker and deployed on **Amazon EKS** with a horizontal pod autoscaler.
- **CI/CD Pipeline**: GitHub Actions automate containerization and deployment of the application.

## How It Works
1. **User Input**: The Streamlit app provides a simple UI where users enter a text sample.
2. **Sentiment Analysis**: Service hosts 
3. **API Processing**: The FastAPI backend processes requests and returns sentiment scores.
4. **Results Visualization**: The frontend displays model outputs side by side for easy comparison.

## API
Source model:  DistilBERT base uncased finetuned SST-2
https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english

API hosts two versions of this model in-memory; 
1) Base model in a huggingface transformers pipeline
2) ONNX version of same model

Each is baked into the docker image and does not need to be downloaded at startup. They are exposed behind two different endpoints, which are utilized by the streamlit application to compare the result and the performance of each model.

Build and run:
```sh
docker build -t fetch-model-service -f docker/Dockerfile .
docker run --rm -p 8080:8080 -t fetch-model-service
```

```sh
# Transformer pipeline
curl -X POST localhost:8080/transformers/infer -H "Content-Type: application/json" -d '{"input": "This is awesome!"}'

# ONNX
curl -X POST localhost:8080/onnx/infer -H "Content-Type: application/json" -d '{"input": "This is awesome!"}'
```

## Streamlit App

Build and run:

```sh
 docker build -t fetch-streamlit -f streamlit_app/Dockerfile ./streamlit_app/
 docker run --rm -p 8501:8501 fetch-streamlit
```

Open your browser and visit localhost:8501.

Note!  This app has a single configuration, which points at the API deployed in EKS.  Edit API->URL in config.ini if you wish to test against an API instance running elsewhere (for example, locally).  

## Kubernetes Ecosystem
![Pods](https://github.com/cbarcroft/fetch-model-service/blob/main/docs/images/kubectl_get_po.PNG)

API:  Min(1) - Max(10) API pods
Streamlit:  1 pod

Horizontal Pod Autoscaler set to 70% CPU threshold
Network Load Balancers for each service

## Deployment Workflow
![Github Actions](https://github.com/cbarcroft/fetch-model-service/blob/main/docs/images/pipeline_overview.PNG)

- **GitHub Actions Workflow**:
  - Builds and pushes Docker images for both the API and Streamlit application.
  - Deploy API to EKS cluster and verify deployment success
  - Deploy configuration for horizontal pod autoscaler and loadbalancer service
  - Deploy Streamlit application to EKS cluster and verify deployment success

Docker build cache has been implemented utilizing ghcr.io registry.  Docker files are layered so that code-only changes will deploy very quickly, and dependency installs are cached between builds unless changed.

## Future Enhancements
Since this was a quick weekend project, obviously there is a lot of room for improvement!
- Implement unit and integration testing with pytest, and integrate into CICD process
- API security using API gateway for rate limiting, and OAuth authentication to prevent unauthorized access
- Use Poetry instead of Pip for better dependency management
- Divide Streamlit app and API into separate repositories.  I combined them here for convenience, but separation would be much cleaner!
- Add stream-based batch processing.  The API and UI are fun to play around with, but they would not be the way to process a real dataset.
