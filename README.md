oerview

This project is an end‑to‑end **movie review sentiment classifier** that predicts whether a given review is **Positive** or **Negative**. The core model is a Simple RNN trained on the [IMDB dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb), which contains 50,000 labeled movie reviews.

The application is wrapped in a **Streamlit web interface**, offering:
- Real‑time prediction with confidence scores
- Review history storage (CSV)
- Batch analysis from PDF files
- Interactive analytics dashboard (sentiment distribution, trends)
- Clean, modern UI with animations

---

##  Key Features

-  **Single Review Analysis** – Type any review and get instant sentiment.
-  **Batch PDF Upload** – Upload a PDF with multiple reviews (separated by blank lines) and analyze all at once.
-  **Review History** – All predictions are saved locally in a CSV file for later review.
-  **Analytics Dashboard** – Visual insights (pie chart, histogram, timeline) based on stored reviews.
-  **Attractive UI** – Gradient backgrounds, animated cards, progress bars, and gauge charts.
-  **Data Export** – Download history or batch results as CSV.
-  **Backward Compatibility** – Automatically handles older CSV formats.

---

##  Technologies Used

| Category          | Tools / Libraries                                     |
|-------------------|-------------------------------------------------------|
| Deep Learning     | TensorFlow, Keras, Simple RNN                         |
| Web Interface     | Streamlit                                             |
| Data Processing   | NumPy, Pandas, re, datetime                           |
| Visualization     | Plotly                                                |
| PDF Extraction    | pdfplumber                                            |
| Dataset           | IMDB (built‑in TensorFlow dataset)                    |

