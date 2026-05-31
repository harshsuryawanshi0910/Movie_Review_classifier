# Import Libraries
import numpy as np
import tensorflow as tf
import re
import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Try to import pdfplumber; if not installed, set to None and handle later
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    pdfplumber = None
    PDF_SUPPORT = False

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
VOCAB_SIZE = 10000
MAX_LEN    = 500
FILE_NAME  = "review_history.csv"

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Sentiment AI | Smart Review Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# FIX 1 – Model & word-index loading
#   • Added compile=False so that the saved .h5 loads even when
#     the custom_objects / optimizer config differs between TF
#     versions (very common cause of load errors).
#   • Word index is loaded ONCE and cached so every prediction
#     call does not re-download it.
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_cached():
    return load_model('simple_rnn_imdb.h5', compile=False)

@st.cache_resource
def load_word_index():
    return imdb.get_word_index()

try:
    word_index = load_word_index()
    model      = load_model_cached()
    # Re-compile with the same settings used during training
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    .custom-card:hover { transform: translateY(-5px); }
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white !important;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        -webkit-text-fill-color: white;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: rgba(255,255,255,0.95) !important;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        color: white;
        font-weight: bold;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        color: white;
        font-weight: bold;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 50px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
    }
    .footer {
        text-align: center;
        color: white !important;
        padding: 1rem;
        margin-top: 2rem;
        border-radius: 10px;
        background: rgba(0,0,0,0.2);
        backdrop-filter: blur(5px);
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    @keyframes fadeInDown {
        from { opacity:0; transform:translateY(-20px); }
        to   { opacity:1; transform:translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity:0; transform:translateY(20px); }
        to   { opacity:1; transform:translateY(0); }
    }
    @keyframes pulse {
        0%,100% { transform:scale(1); }
        50%      { transform:scale(1.05); }
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FIX 2 – Preprocessing
#   Original bug: word_index values are 1-based, but the IMDB
#   loader reserves indices 0 (padding), 1 (start), 2 (unknown),
#   3 (unused).  The correct offset is +3, BUT only when the
#   result is < VOCAB_SIZE; otherwise fall back to the OOV index
#   (index 2).  Also strip punctuation BEFORE lowercasing so
#   contractions like "don't" split cleanly.
# ─────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> np.ndarray:
    text  = re.sub(r"[^a-zA-Z\s]", "", text)   # keep only letters + spaces
    words = text.lower().split()

    encoded = []
    for word in words:
        # word_index is 1-based; IMDB protocol adds 3 to each index
        idx = word_index.get(word, 0)
        if idx == 0:
            encoded.append(2)          # unknown token
        else:
            idx_shifted = idx + 3
            encoded.append(idx_shifted if idx_shifted < VOCAB_SIZE else 2)

    padded = sequence.pad_sequences([encoded], maxlen=MAX_LEN, padding='pre')
    return padded

# ─────────────────────────────────────────────────────────────
# FIX 3 – CSV helpers
#   • strip() reviews before saving to avoid trailing \n chars.
#   • Use a safe read with encoding fallback.
# ─────────────────────────────────────────────────────────────
def save_review(review: str, sentiment: str, score: float):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_new = pd.DataFrame({
        "Timestamp":        [timestamp],
        "Review":           [review.strip()],
        "Sentiment":        [sentiment],
        "Confidence Score": [round(float(score), 6)],
    })

    if os.path.exists(FILE_NAME):
        try:
            existing = pd.read_csv(FILE_NAME)
            # backward-compat rename
            if 'Score' in existing.columns and 'Confidence Score' not in existing.columns:
                existing = existing.rename(columns={'Score': 'Confidence Score'})
                existing.to_csv(FILE_NAME, index=False)
            df_new.to_csv(FILE_NAME, mode='a', header=False, index=False)
        except Exception:
            df_new.to_csv(FILE_NAME, index=False)
    else:
        df_new.to_csv(FILE_NAME, index=False)


def load_history() -> pd.DataFrame:
    if os.path.exists(FILE_NAME):
        try:
            df = pd.read_csv(FILE_NAME, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(FILE_NAME, encoding='latin-1')
        # Normalise column names
        if 'Score' in df.columns and 'Confidence Score' not in df.columns:
            df = df.rename(columns={'Score': 'Confidence Score'})
        if 'Timestamp' not in df.columns:
            df.insert(0, 'Timestamp', '')
        # Strip accidental whitespace from the Review column
        if 'Review' in df.columns:
            df['Review'] = df['Review'].astype(str).str.strip()
        return df
    return pd.DataFrame(columns=["Timestamp", "Review", "Sentiment", "Confidence Score"])


def get_statistics(df: pd.DataFrame):
    if len(df) == 0:
        return 0, 0, 0, 0.0
    total    = len(df)
    positive = int((df['Sentiment'] == 'Positive').sum())
    negative = int((df['Sentiment'] == 'Negative').sum())
    score_col = 'Confidence Score' if 'Confidence Score' in df.columns else 'Score'
    avg_conf  = float(df[score_col].mean()) if score_col in df.columns else 0.0
    return total, positive, negative, avg_conf

# ─────────────────────────────────────────────────────────────
# FIX 4 – PDF helpers
#   Added a guard so we return "" (not crash) when no text found.
# ─────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_file) -> str:
    if not PDF_SUPPORT:
        return ""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def split_into_reviews(text: str, method: str = 'blank_lines') -> list:
    text = re.sub(r'\r\n', '\n', text)
    if method == 'blank_lines':
        reviews = re.split(r'\n\s*\n', text.strip())
    else:
        reviews = text.split('\n')
    return [r.strip() for r in reviews if r.strip()]

# ─────────────────────────────────────────────────────────────
# FIX 5 – Batch prediction helper
#   Running model.predict() in a loop on single samples is very
#   slow because TF rebuilds the computation graph each call.
#   Batch all sequences first, predict once.
# ─────────────────────────────────────────────────────────────
def batch_predict(reviews: list) -> list:
    """Return list of (sentiment, confidence, raw_score) tuples."""
    sequences = np.vstack([preprocess_text(r) for r in reviews])   # (N, MAX_LEN)
    preds     = model.predict(sequences, batch_size=32, verbose=0).flatten()

    results = []
    for p in preds:
        p = float(p)
        sentiment  = "Positive" if p > 0.5 else "Negative"
        confidence = p if p > 0.5 else 1 - p
        results.append((sentiment, float(confidence), p))
    return results

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="text-align:center;padding:1rem;">
            <h2 style="color:white;">🎬 Movie Sentiment AI</h2>
            <p style="color:#9ca3af;">Your Smart Review Analyzer</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    history_df                          = load_history()
    total, positive, negative, avg_conf = get_statistics(history_df)

    st.markdown("### 📊 Statistics")
    c1, c2 = st.columns(2)
    c1.metric("Total Reviews", total)
    c1.metric("Positive",      positive)
    c2.metric("Negative",      negative)
    c2.metric("Avg Confidence", f"{avg_conf:.1%}" if avg_conf > 0 else "N/A")

    st.markdown("---")
    st.markdown("### About")
    st.info("This AI model analyzes movie reviews and determines whether they are positive or negative. "
            "Built with TensorFlow and trained on the IMDB dataset.")
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.success("• Write detailed reviews for better accuracy\n"
               "• Include specific examples\n"
               "• Mention what you liked / disliked")

# ─────────────────────────────────────────────────────────────
# Main header
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 Movie Review Sentiment AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Deep Learning • Analyze your movie reviews in seconds</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Analyze Review", "📋 Review History", "📊 Analytics", "📄 Batch Upload (PDF)"]
)

# ─────────────────────────────────────────────────────────────
# Tab 1 – Single review
# ─────────────────────────────────────────────────────────────
with tab1:
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### ✍️ Enter Your Review")

        if 'review_text' not in st.session_state:
            st.session_state.review_text = ""

        st.markdown("**Need inspiration? Try these examples:**")
        ex1, ex2 = st.columns(2)
        if ex1.button("😀 Positive Example", key="positive_example"):
            st.session_state.review_text = (
                "This movie is a masterpiece! The cinematography was breathtaking and the "
                "performances were outstanding. I was completely captivated from start to finish!"
            )
        if ex2.button("😞 Negative Example", key="negative_example"):
            st.session_state.review_text = (
                "What a waste of time! The plot was predictable, the acting was wooden, "
                "and the special effects looked like they were from 20 years ago."
            )

        user_input = st.text_area(
            "What did you think about the movie?",
            value=st.session_state.review_text,
            height=150,
            placeholder="Example: This movie was absolutely fantastic! The acting was superb...",
            key="review_input"
        )
        st.session_state.review_text = user_input

        st.markdown("---")
        if st.button("🔍 Analyze Sentiment", use_container_width=True, key="classify_btn"):
            if not user_input.strip():
                st.warning("⚠️ Please enter a review to analyze.")
            else:
                with st.spinner("Analyzing your review…"):
                    processed  = preprocess_text(user_input)
                    raw_score  = float(model.predict(processed, verbose=0)[0][0])
                    sentiment  = "Positive" if raw_score > 0.5 else "Negative"
                    confidence = raw_score if raw_score > 0.5 else 1 - raw_score

                    save_review(user_input, sentiment, raw_score)

                st.markdown("### 🎯 Analysis Results")
                badge_class = "sentiment-positive" if sentiment == "Positive" else "sentiment-negative"
                icon        = "✅" if sentiment == "Positive" else "❌"
                st.markdown(
                    f'<div class="{badge_class}" style="margin:0 auto;text-align:center;">'
                    f'{icon} {sentiment} Sentiment</div>',
                    unsafe_allow_html=True
                )
                st.markdown(f"**Confidence:** {confidence:.1%}")
                st.progress(float(confidence))

                fig = go.Figure(go.Indicator(
                    mode  = "gauge+number",
                    value = raw_score * 100,
                    title = {'text': "Sentiment Score"},
                    domain= {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis':      {'range': [0, 100]},
                        'bar':       {'color': "#667eea"},
                        'steps':     [{'range': [0, 50],  'color': "#fee2e2"},
                                      {'range': [50, 100], 'color': "#d1fae5"}],
                        'threshold': {'line': {'color': "red", 'width': 4},
                                      'thickness': 0.75, 'value': 50}
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Tab 2 – History
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Review History")

    history = load_history()

    if len(history) == 0:
        st.info("No reviews analyzed yet. Start by analyzing a review in the 'Analyze Review' tab!")
    else:
        display_df = history.copy()
        if 'Confidence Score' in display_df.columns:
            display_df['Confidence Score'] = display_df['Confidence Score'].apply(lambda x: f"{float(x):.4f}")

        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Timestamp":        st.column_config.TextColumn("Time"),
                "Review":           st.column_config.TextColumn("Review", width="large"),
                "Sentiment":        st.column_config.TextColumn("Sentiment"),
                "Confidence Score": st.column_config.TextColumn("Confidence"),
            }
        )

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                label     = "⬇️ Download as CSV",
                data      = history.to_csv(index=False),
                file_name = f"movie_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime      = "text/csv",
                use_container_width=True,
                key       = "download_btn"
            )
        with c2:
            if st.button("🗑️ Clear History", use_container_width=True, key="clear_btn"):
                if os.path.exists(FILE_NAME):
                    os.remove(FILE_NAME)
                    st.success("History cleared successfully!")
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Tab 3 – Analytics
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Analytics Dashboard")

    history = load_history()

    if len(history) > 0:
        if 'Score' in history.columns and 'Confidence Score' not in history.columns:
            history = history.rename(columns={'Score': 'Confidence Score'})

        c1, c2 = st.columns(2)
        with c1:
            sentiment_counts = history['Sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_sequence=['#10b981', '#ef4444']
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            if 'Confidence Score' in history.columns:
                fig_hist = px.histogram(
                    history, x='Confidence Score',
                    title="Confidence Score Distribution",
                    color='Sentiment',
                    nbins=20,
                    color_discrete_map={'Positive': '#10b981', 'Negative': '#ef4444'}
                )
                fig_hist.update_layout(bargap=0.1)
                st.plotly_chart(fig_hist, use_container_width=True)

        # Timeline
        if len(history) > 1 and 'Timestamp' in history.columns:
            try:
                history['Timestamp'] = pd.to_datetime(history['Timestamp'], errors='coerce')
                history_sorted = history.dropna(subset=['Timestamp']).sort_values('Timestamp')
                fig_line = px.line(
                    history_sorted, x='Timestamp', y='Confidence Score',
                    title="Sentiment Trend Over Time",
                    color='Sentiment',
                    markers=True,
                    color_discrete_map={'Positive': '#10b981', 'Negative': '#ef4444'}
                )
                st.plotly_chart(fig_line, use_container_width=True)
            except Exception:
                st.warning("Unable to create timeline chart.")

        st.markdown("### 📈 Key Metrics")
        total, positive, negative, avg_conf = get_statistics(history)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Reviews", total)
        m2.metric("Positive Reviews", positive,
                  delta=f"{positive/total*100:.0f}%" if total else None)
        m3.metric("Negative Reviews", negative,
                  delta=f"-{negative/total*100:.0f}%" if total else None)
        m4.metric("Average Confidence", f"{avg_conf:.1%}" if avg_conf > 0 else "N/A")
    else:
        st.info("No data available yet. Start analyzing reviews to see analytics!")

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Tab 4 – Batch PDF
# ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📄 Batch Analyze Reviews from PDF")
    st.markdown("Upload a PDF containing multiple movie reviews and analyze all at once.")

    if not PDF_SUPPORT:
        st.error("The 'pdfplumber' library is not installed. Run: `pip install pdfplumber`")
    else:
        method = st.radio(
            "Review separation method",
            options=["Blank lines (multiline reviews)", "Each line as separate review"],
            help="Choose 'Blank lines' if reviews are paragraphs separated by empty lines."
        )
        method_value  = 'blank_lines' if "Blank lines" in method else 'each_line'
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")

        if uploaded_file is not None:
            try:
                with st.spinner("Extracting text from PDF…"):
                    pdf_text = extract_text_from_pdf(uploaded_file)

                if not pdf_text.strip():
                    st.warning("No text found in the PDF. Please ensure the PDF contains selectable text.")
                else:
                    reviews = split_into_reviews(pdf_text, method=method_value)
                    st.info(f"Found **{len(reviews)}** reviews to analyze.")

                    with st.expander("Preview extracted reviews"):
                        for i, rev in enumerate(reviews[:5]):
                            st.write(f"**Review {i+1}:**")
                            st.write(rev[:200] + "…" if len(rev) > 200 else rev)
                        if len(reviews) > 5:
                            st.write(f"… and {len(reviews)-5} more.")

                    if st.button("🚀 Analyze All Reviews", key="batch_analyze_btn"):
                        if not reviews:
                            st.warning("No reviews to analyze.")
                        else:
                            with st.spinner(f"Analyzing {len(reviews)} reviews…"):
                                # FIX 5 in action: one batched predict call
                                predictions = batch_predict(reviews)

                            results = []
                            for review, (sentiment, confidence, raw_score) in zip(reviews, predictions):
                                save_review(review, sentiment, raw_score)
                                results.append({
                                    "Review":           review,
                                    "Sentiment":        sentiment,
                                    "Confidence Score": round(confidence, 4),
                                    "Prediction Score": round(raw_score,  4),
                                })

                            df_results = pd.DataFrame(results)
                            st.success("✅ Analysis complete!")
                            st.markdown("### 📋 Batch Results")

                            r1, r2, r3 = st.columns(3)
                            r1.metric("Total Reviews",    len(df_results))
                            r2.metric("Positive",         int((df_results['Sentiment'] == 'Positive').sum()))
                            r3.metric("Negative",         int((df_results['Sentiment'] == 'Negative').sum()))

                            st.dataframe(df_results, use_container_width=True,
                                         column_config={
                                             "Review":           st.column_config.TextColumn("Review", width="large"),
                                             "Sentiment":        st.column_config.TextColumn("Sentiment"),
                                             "Confidence Score": st.column_config.NumberColumn("Confidence", format="%.2f"),
                                             "Prediction Score": st.column_config.NumberColumn("Score",      format="%.4f"),
                                         })

                            st.download_button(
                                label     = "⬇️ Download Results as CSV",
                                data      = df_results.to_csv(index=False),
                                file_name = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime      = "text/csv",
                                key       = "batch_download"
                            )
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("""
    <div class="footer">
         © 2026 Movie Sentiment AI
    </div>
""", unsafe_allow_html=True)