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

# Config
VOCAB_SIZE = 10000
MAX_LEN = 500
FILE_NAME = "review_history.csv"

# Page Configuration
st.set_page_config(
    page_title="Movie Sentiment AI | Smart Review Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Data
@st.cache_resource
def load_model_cached():
    return load_model('simple_rnn_imdb.h5')

@st.cache_data
def load_word_index():
    return imdb.get_word_index()

try:
    word_index = load_word_index()
    model = load_model_cached()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Custom CSS for Enhanced UI (with visible titles and footer)
st.markdown("""
    <style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
    }
    
    /* Title styling - improved visibility */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white !important;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        background: none;
        -webkit-text-fill-color: white;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.95) !important;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Ensure headings inside cards are readable */
    .custom-card h1, 
    .custom-card h2, 
    .custom-card h3, 
    .custom-card h4,
    .custom-card .stMarkdown h1,
    .custom-card .stMarkdown h2,
    .custom-card .stMarkdown h3 {
        color: #1f2937 !important;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: white !important;
        padding: 1rem;
        margin-top: 2rem;
        border-radius: 10px;
        background: rgba(0,0,0,0.2);
        backdrop-filter: blur(5px);
    }
    
    /* Sentiment badges */
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
    
    /* Button styling */
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
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Metrics styling */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Success message styling */
    .stAlert {
        border-radius: 15px;
        border-left: 4px solid #10b981;
    }
    </style>
""", unsafe_allow_html=True)

# Preprocessing Function
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

# Save to CSV with timestamp
def save_review(review, sentiment, score):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "Timestamp": [timestamp],
        "Review": [review],
        "Sentiment": [sentiment],
        "Confidence Score": [score]
    }
    
    df = pd.DataFrame(data)
    
    if os.path.exists(FILE_NAME):
        try:
            existing_df = pd.read_csv(FILE_NAME)
            if 'Score' in existing_df.columns and 'Confidence Score' not in existing_df.columns:
                existing_df = existing_df.rename(columns={'Score': 'Confidence Score'})
                existing_df.to_csv(FILE_NAME, index=False)
            df.to_csv(FILE_NAME, mode='a', header=False, index=False)
        except:
            df.to_csv(FILE_NAME, index=False)
    else:
        df.to_csv(FILE_NAME, index=False)

# Load History with backward compatibility
def load_history():
    if os.path.exists(FILE_NAME):
        df = pd.read_csv(FILE_NAME)
        if 'Score' in df.columns and 'Confidence Score' not in df.columns:
            df = df.rename(columns={'Score': 'Confidence Score'})
        if 'Timestamp' not in df.columns:
            df.insert(0, 'Timestamp', '')
        return df
    else:
        return pd.DataFrame(columns=["Timestamp", "Review", "Sentiment", "Confidence Score"])

# Get statistics with error handling
def get_statistics(df):
    if len(df) > 0:
        total_reviews = len(df)
        positive_reviews = len(df[df['Sentiment'] == 'Positive'])
        negative_reviews = len(df[df['Sentiment'] == 'Negative'])
        
        if 'Confidence Score' in df.columns:
            avg_confidence = df['Confidence Score'].mean()
        elif 'Score' in df.columns:
            avg_confidence = df['Score'].mean()
        else:
            avg_confidence = 0
            
        return total_reviews, positive_reviews, negative_reviews, avg_confidence
    return 0, 0, 0, 0

# Function to extract text from PDF (only if pdfplumber is available)
def extract_text_from_pdf(pdf_file):
    if not PDF_SUPPORT:
        return None
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: white;">🎬 Movie Sentiment AI</h2>
            <p style="color: #9ca3af;">Your Smart Review Analyzer</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    history_df = load_history()
    total, positive, negative, avg_conf = get_statistics(history_df)
    
    st.markdown("### 📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Reviews", total)
        st.metric("Positive", positive)
    with col2:
        st.metric("Negative", negative)
        st.metric("Avg Confidence", f"{avg_conf:.1%}" if avg_conf > 0 else "N/A")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
        This AI model analyzes movie reviews and determines if they're positive or negative.
        Built with TensorFlow and trained on the IMDB dataset.
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Tips")
    st.success("""
        • Write detailed reviews for better accuracy
        • Include specific examples
        • Mention what you liked/disliked
    """)

# Main Content
st.markdown('<div class="main-title">🎬 Movie Review Sentiment AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Deep Learning • Analyze your movie reviews in seconds</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze Review", "📜 Review History", "📈 Analytics", "📄 Batch Upload (PDF)"])

# Tab 1: Single review analysis
with tab1:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### ✍️ Enter Your Review")
        
        if 'review_text' not in st.session_state:
            st.session_state.review_text = ""
        
        st.markdown("**Need inspiration? Try these examples:**")
        ex_col1, ex_col2 = st.columns(2)
        
        if ex_col1.button("🎭 Positive Example", key="positive_example"):
            st.session_state.review_text = "This movie is a masterpiece! The cinematography was breathtaking and the performances were outstanding. I was completely captivated from start to finish!"
        
        if ex_col2.button("💔 Negative Example", key="negative_example"):
            st.session_state.review_text = "What a waste of time! The plot was predictable, the acting was wooden, and the special effects looked like they were from 20 years ago."
        
        user_input = st.text_area(
            "What did you think about the movie?",
            value=st.session_state.review_text,
            height=150,
            placeholder="Example: This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout...",
            key="review_input"
        )
        
        st.session_state.review_text = user_input
        
        st.markdown("---")
        
        classify_btn = st.button("🔍 Analyze Sentiment", use_container_width=True, key="classify_btn")
        
        if classify_btn:
            if user_input.strip() == "":
                st.warning("⚠️ Please enter a review to analyze")
            else:
                with st.spinner("Analyzing your review..."):
                    processed = preprocess_text(user_input)
                    prediction = model.predict(processed)[0][0]
                    prediction = float(prediction)
                    sentiment = "Positive" if prediction > 0.5 else "Negative"
                    confidence = prediction if prediction > 0.5 else 1 - prediction
                    confidence = float(confidence)
                    
                    save_review(user_input, sentiment, prediction)
                    
                    st.markdown("### 📊 Analysis Results")
                    
                    if sentiment == "Positive":
                        st.markdown(f'<div class="sentiment-positive" style="margin: 0 auto; text-align: center;">🎉 {sentiment} Sentiment</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="sentiment-negative" style="margin: 0 auto; text-align: center;">😞 {sentiment} Sentiment</div>', unsafe_allow_html=True)
                    
                    st.markdown(f"**Confidence:** {confidence:.1%}")
                    st.progress(confidence)
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction * 100,
                        title = {'text': "Sentiment Score"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#667eea"},
                            'steps': [
                                {'range': [0, 50], 'color': "#fee2e2"},
                                {'range': [50, 100], 'color': "#d1fae5"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: History
with tab2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📜 Review History")
    
    history = load_history()
    
    if len(history) == 0:
        st.info("No reviews analyzed yet. Start by analyzing a review in the 'Analyze Review' tab!")
    else:
        display_df = history.copy()
        
        if 'Confidence Score' in display_df.columns:
            display_df['Confidence Score'] = display_df['Confidence Score'].apply(lambda x: f"{x:.4f}")
        elif 'Score' in display_df.columns:
            display_df = display_df.rename(columns={'Score': 'Confidence Score'})
            display_df['Confidence Score'] = display_df['Confidence Score'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Timestamp": st.column_config.TextColumn("Time"),
                "Review": st.column_config.TextColumn("Review", width="large"),
                "Sentiment": st.column_config.TextColumn("Sentiment"),
                "Confidence Score": st.column_config.TextColumn("Confidence")
            }
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download as CSV",
                data=history.to_csv(index=False),
                file_name=f"movie_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_btn"
            )
        
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True, key="clear_btn"):
                if os.path.exists(FILE_NAME):
                    os.remove(FILE_NAME)
                    st.success("History cleared successfully!")
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Analytics
with tab3:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Analytics Dashboard")
    
    history = load_history()
    
    if len(history) > 0:
        if 'Score' in history.columns and 'Confidence Score' not in history.columns:
            history = history.rename(columns={'Score': 'Confidence Score'})
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Sentiment' in history.columns:
                sentiment_counts = history['Sentiment'].value_counts()
                fig_pie = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color_discrete_sequence=['#10b981', '#ef4444']
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            if 'Confidence Score' in history.columns:
                fig_hist = px.histogram(
                    history,
                    x='Confidence Score',
                    title="Confidence Score Distribution",
                    color='Sentiment' if 'Sentiment' in history.columns else None,
                    nbins=20,
                    color_discrete_sequence=['#10b981', '#ef4444']
                )
                fig_hist.update_layout(bargap=0.1)
                st.plotly_chart(fig_hist, use_container_width=True)
        
        if len(history) > 1 and 'Timestamp' in history.columns:
            try:
                history['Timestamp'] = pd.to_datetime(history['Timestamp'])
                history_sorted = history.sort_values('Timestamp')
                
                fig_line = px.line(
                    history_sorted,
                    x='Timestamp',
                    y='Confidence Score',
                    title="Sentiment Trend Over Time",
                    color='Sentiment' if 'Sentiment' in history.columns else None,
                    markers=True
                )
                fig_line.update_layout(xaxis_title="Time", yaxis_title="Confidence Score")
                st.plotly_chart(fig_line, use_container_width=True)
            except:
                st.warning("Unable to create timeline chart. Ensure timestamps are in correct format.")
        
        st.markdown("### 📊 Key Metrics")
        total, positive, negative, avg_conf = get_statistics(history)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Reviews", total)
        with col2:
            if total > 0:
                st.metric("Positive Reviews", positive, delta=f"{positive/total*100:.0f}%")
            else:
                st.metric("Positive Reviews", positive)
        with col3:
            if total > 0:
                st.metric("Negative Reviews", negative, delta=f"-{negative/total*100:.0f}%")
            else:
                st.metric("Negative Reviews", negative)
        with col4:
            st.metric("Average Confidence", f"{avg_conf:.1%}" if avg_conf > 0 else "N/A")
    
    else:
        st.info("No data available yet. Start analyzing reviews to see analytics!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 4: Batch Upload PDF (with blank line separator)
with tab4:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📄 Batch Analyze Reviews from PDF")
    st.markdown("Upload a PDF file containing multiple movie reviews. The app will extract each review and analyze sentiment.")
    
    if not PDF_SUPPORT:
        st.error("❌ The 'pdfplumber' library is not installed. To use this feature, please install it by running: `pip install pdfplumber`")
        st.stop()
    
    # Select separation method
    st.markdown("**Select how reviews are separated in the PDF:**")
    method = st.radio(
        "Review separation method",
        options=["Blank lines (paragraphs)", "Each line as separate review"],
        help="If reviews are separated by blank lines (paragraphs), choose the first option. If each review is on its own line, choose the second."
    )
    method_value = 'blank_lines' if method == "Blank lines (paragraphs)" else 'each_line'
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
    
    if uploaded_file is not None:
        try:
            with st.spinner("Extracting text from PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
            
            if not pdf_text.strip():
                st.warning("No text found in the PDF. Please ensure the PDF contains selectable text.")
            else:
                # Split based on method
                if method_value == 'blank_lines':
                    # Split by blank lines (one or more newlines)
                    import re
                    # Normalize newlines
                    text = re.sub(r'\r\n', '\n', pdf_text)
                    # Split on blank lines (one or more empty lines)
                    reviews = re.split(r'\n\s*\n', text.strip())
                    # Remove empty strings and strip each review
                    reviews = [r.strip() for r in reviews if r.strip()]
                else:
                    # Each line
                    reviews = [line.strip() for line in pdf_text.split('\n') if line.strip()]
                
                st.info(f"Found {len(reviews)} reviews to analyze.")
                
                # Preview extracted reviews
                with st.expander("Preview extracted reviews"):
                    for i, rev in enumerate(reviews[:5]):
                        st.write(f"**Review {i+1}:**")
                        st.write(rev[:200] + "..." if len(rev) > 200 else rev)
                    if len(reviews) > 5:
                        st.write(f"... and {len(reviews)-5} more.")
                
                if st.button("🚀 Analyze All Reviews", key="batch_analyze_btn"):
                    if len(reviews) == 0:
                        st.warning("No reviews to analyze.")
                    else:
                        results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, review in enumerate(reviews):
                            progress = (i + 1) / len(reviews)
                            progress_bar.progress(progress)
                            status_text.text(f"Analyzing review {i+1} of {len(reviews)}...")
                            
                            processed = preprocess_text(review)
                            prediction = model.predict(processed)[0][0]
                            prediction = float(prediction)
                            sentiment = "Positive" if prediction > 0.5 else "Negative"
                            confidence = prediction if prediction > 0.5 else 1 - prediction
                            confidence = float(confidence)
                            
                            results.append({
                                "Review": review,
                                "Sentiment": sentiment,
                                "Confidence Score": confidence,
                                "Prediction Score": prediction
                            })
                            
                            save_review(review, sentiment, prediction)
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        df_results = pd.DataFrame(results)
                        
                        st.success("Analysis complete!")
                        st.markdown("### 📊 Batch Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Reviews", len(df_results))
                        with col2:
                            pos_count = len(df_results[df_results['Sentiment'] == 'Positive'])
                            st.metric("Positive", pos_count)
                        with col3:
                            neg_count = len(df_results[df_results['Sentiment'] == 'Negative'])
                            st.metric("Negative", neg_count)
                        
                        st.dataframe(
                            df_results,
                            use_container_width=True,
                            column_config={
                                "Review": st.column_config.TextColumn("Review", width="large"),
                                "Sentiment": st.column_config.TextColumn("Sentiment"),
                                "Confidence Score": st.column_config.NumberColumn("Confidence", format="%.2f"),
                                "Prediction Score": st.column_config.NumberColumn("Score", format="%.4f")
                            }
                        )
                        
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="batch_download"
                        )
        
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        Made with  using TensorFlow & Streamlit | © 2024 Movie Sentiment AI
    </div>
""", unsafe_allow_html=True)