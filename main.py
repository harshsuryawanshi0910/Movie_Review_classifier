# ================================
# Import Libraries
# ================================
import numpy as np
import tensorflow as tf
import re
import streamlit as st
import pandas as pd
import os

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# ================================
# Config
# ================================
VOCAB_SIZE = 10000
MAX_LEN = 500
FILE_NAME = "review_history.csv"

# ================================
# Load Data
# ================================
word_index = imdb.get_word_index()
model = load_model('simple_rnn_imdb.h5')

# ================================
# Preprocessing Function
# ================================
def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text)
    words = text.lower().split()

    encoded_review = []
    for word in words:
        index = word_index.get(word, 2) + 3
        if index >= VOCAB_SIZE:
            index = 2
        encoded_review.append(index)

    padded_review = sequence.pad_sequences([encoded_review], maxlen=MAX_LEN)
    return padded_review

# ================================
# Save to CSV
# ================================
def save_review(review, sentiment, score):
    data = {
        "Review": [review],
        "Sentiment": [sentiment],
        "Score": [score]
    }

    df = pd.DataFrame(data)

    if os.path.exists(FILE_NAME):
        df.to_csv(FILE_NAME, mode='a', header=False, index=False)
    else:
        df.to_csv(FILE_NAME, index=False)

# ================================
# Load History
# ================================
def load_history():
    if os.path.exists(FILE_NAME):
        return pd.read_csv(FILE_NAME)
    else:
        return pd.DataFrame(columns=["Review", "Sentiment", "Score"])

# ================================
#  Streamlit UI Design
# ================================
st.set_page_config(page_title="Movie Sentiment AI", page_icon="🎬", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #ff4b4b;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #bbbbbb;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 45px;
        width: 150px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title"> Movie Review Sentiment AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze your movie reviews instantly </div>', unsafe_allow_html=True)

st.write("")

# Input Box
user_input = st.text_area(" Enter your review", height=150)

# Buttons
col1, col2 = st.columns(2)

with col1:
    classify_btn = st.button("🔍 Classify")

with col2:
    show_history_btn = st.button("Show History")

# ================================
# Classification
# ================================
if classify_btn:
    if user_input.strip() == "":
        st.warning(" Please enter a review")
    else:
        processed = preprocess_text(user_input)

        prediction = model.predict(processed)[0][0]
        sentiment = "Positive " if prediction > 0.5 else "Negative "

        # Save review
        save_review(user_input, sentiment, float(prediction))

        # Display result
        st.success(f" Sentiment: {sentiment}")
        st.info(f"Confidence Score: {prediction:.4f}")

        # Progress bar
        st.progress(int(prediction * 100))

# ================================
# Show History
# ================================
if show_history_btn:
    st.subheader(" Review History")

    history = load_history()

    if len(history) == 0:
        st.write("No history found.")
    else:
        st.dataframe(history)

        # Download option
        st.download_button(
            label=" Download History",
            data=history.to_csv(index=False),
            file_name="review_history.csv",
            mime="text/csv"
        )