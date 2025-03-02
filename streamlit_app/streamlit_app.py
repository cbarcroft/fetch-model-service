import streamlit as st
import requests
import time
import pandas as pd

API_URL = "http://k8s-default-fetchmod-7a9fee2176-0ade04e7460449c5.elb.us-west-2.amazonaws.com/infer"

st.title("Sentiment Analysis with DistilBERT")

st.sidebar.title("App Info")
st.sidebar.write("This app uses a fine-tuned DistilBERT model for sentiment analysis.")

user_input = st.text_area("Enter text for sentiment analysis:")

@st.cache_data
def fetch_sentiment(text):
    response = requests.post(API_URL, json={"input": text})
    if response.status_code == 200:
        return response.json()
    return None

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Analyze Sentiment"):
    with st.spinner("Analyzing..."):
        start_time = time.time()
        result = fetch_sentiment(user_input)
        end_time = time.time()
    
    response_time = round((end_time - start_time) * 1000, 2)

    if result:
        sentiment_label = result[0]["label"].upper()
        sentiment_score = result[0]["score"]

        sentiment_colors = {
            "POSITIVE": "🟢 Positive",
            "NEGATIVE": "🔴 Negative",
            "NEUTRAL": "🟡 Neutral"
        }
        
        st.markdown(f"### Sentiment: {sentiment_colors.get(sentiment_label, '⚪ Unknown')}")
        st.metric(label="Sentiment Score", value=f"{sentiment_score:.2f}")
        st.metric(label="Response Time", value=f"{response_time} ms")

        st.session_state.history.append({
            "Text": user_input, 
            "Sentiment": sentiment_label, 
            "Score": sentiment_score, 
            "Response Time (ms)": response_time
        })

    else:
        st.error("API request failed.")

if st.session_state.history:
    st.write("### Analysis History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)
