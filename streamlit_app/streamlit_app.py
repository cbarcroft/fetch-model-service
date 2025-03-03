import streamlit as st
import requests
import time
import pandas as pd

import configparser

config = configparser.ConfigParser()
config.read("../config.ini")

# API Endpoints
BASE_URL = config["api"]["url"]
TRANSFORMERS_API_URL = f"{BASE_URL}/transformers/infer"
ONNX_API_URL = f"{BASE_URL}/onnx/infer"

st.set_page_config(layout="wide")

st.title("Sentiment Analysis Performance: Transformer vs. ONNX")

st.sidebar.title("App Info")
st.sidebar.write("Compare sentiment analysis performance between Huggingface Transformer and ONNX models.")
st.sidebar.write("Source Model:  https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Call into one of the model APIs
def fetch_sentiment(api_url, text):
    response = requests.post(api_url, json={"input": text})
    if response.status_code == 200:
        return response.json()
    return None

def process_sentiment_analysis(api_url, model_name, col):
    """Given a streamlit column, perform inference using the indicated model and display result."""

    with col:
        st.subheader(f"{model_name} Model")
        with st.spinner(f"Analyzing with {model_name}..."):
            start_time = time.time()
            result = fetch_sentiment(api_url, user_input)
            end_time = time.time()

        response_time = round((end_time - start_time) * 1000, 2)

        if result:
            sentiment_label = result[0]["label"].upper() if model_name == "Transformers" else result["sentiment"].upper()
            sentiment_score = result[0]["score"] if model_name == "Transformers" else result["score"]

            # User friendly color-coded icon labels
            sentiment_colors = {
                "POSITIVE": "🟢 Positive",
                "NEGATIVE": "🔴 Negative",
                "NEUTRAL": "🟡 Neutral"
            }

            st.markdown(f"### Sentiment: {sentiment_colors.get(sentiment_label, '⚪ Unknown')}")
            st.metric(label="Sentiment Score", value=f"{sentiment_score:.2f}")
            st.metric(label="Response Time", value=f"{response_time} ms")

            return {
                f"{model_name} Sentiment": sentiment_label,
                f"{model_name} Score": sentiment_score,
                f"{model_name} Response Time (ms)": response_time
            }
        else:
            st.error(f"{model_name} API request failed.")
            return None

# Initialize session state for tracking call history
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_area("Enter text for sentiment analysis:")

if st.button("Submit"):
    if user_input.strip():
        transformer_col, onnx_col = st.columns(2)

        # Run analysis for each model in its own side by side aligned column.
        transformers_data = process_sentiment_analysis(TRANSFORMERS_API_URL, "Transformers", transformer_col)
        onnx_data = process_sentiment_analysis(ONNX_API_URL, "ONNX", onnx_col)

        # Store history in session state if both results exist
        if transformers_data and onnx_data:
            st.session_state.history.append({
                "Text": user_input, 
                **transformers_data,
                **onnx_data,
                "Response Time Difference (%)": round(((onnx_data["ONNX Response Time (ms)"] - transformers_data["Transformers Response Time (ms)"]) / transformers_data["Transformers Response Time (ms)"]) * 100, 2) 
                if transformers_data["Transformers Response Time (ms)"] != 0 else 0
            })
    else:
        st.warning("Please enter some text before analyzing.")

#Render history table
if st.session_state.history:
    st.write("### Analysis History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)
