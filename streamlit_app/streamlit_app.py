import streamlit as st
import requests
import time

API_URL = "http://k8s-default-fetchmod-8ad9a1129e-d49f1c596431639a.elb.us-west-2.amazonaws.com/infer"

st.title("Sentiment Analysis with Distilbert")

user_input = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze Sentiment"):
    start_time = time.time()
    response = requests.post(API_URL, json={"input": user_input})
    end_time = time.time()
    response_time = round((end_time - start_time) * 1000, 2)

    if response.status_code == 200:
        result = response.json()
        st.write(f"### Sentiment: {result[0]['label']}")
        st.write(f"### Sentiment Score: {result[0]['score']}")
        st.write(f"#### Response Time: {response_time} ms")
    else:
        st.write("Error:", response.text)