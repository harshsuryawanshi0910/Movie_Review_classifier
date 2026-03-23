# movie review classifier using simple RNN #
Project Summary

This project is a Movie Review Sentiment Classifier built using a Simple Recurrent Neural Network (RNN). The model analyzes textual movie reviews and classifies them as Positive or Negative based on their sentiment.

The system is trained on the IMDB dataset, which contains thousands of labeled movie reviews. By learning patterns in sequences of words, the RNN model captures contextual meaning and predicts sentiment effectively.


A user-friendly Streamlit web interface is integrated, allowing users to enter their own movie reviews and get instant predictions along with confidence scores. Additionally, the application stores previously classified reviews in a CSV file for future reference and analysis.


# Key Features

 1.Deep Learning Model: Built using Simple RNN for sequence processing

 2.Sentiment Classification: Predicts Positive or Negative reviews

 3.Real-time Prediction: Instant results via Streamlit interface

 4.Review History Storage: Saves all predictions in a CSV file

 5.Interactive UI: Clean and attractive user interface

 6.Text Preprocessing: Tokenization, padding, and vocabulary control



# Technologies Used

Python

TensorFlow / Keras

Streamlit

NumPy & Pandas

IMDB Dataset



# How It Works

1.User enters a movie review in the web app

2.Text is preprocessed (cleaning, tokenization, padding)

3.Converted into numerical sequence using IMDB word index

4.Passed into the trained Simple RNN model

5.Model predicts sentiment score

6.Result displayed as Positive or Negative

7.Review is stored in a CSV file

