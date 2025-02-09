from flask import Flask, request, jsonify
import joblib
import numpy as np
import nltk
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load trained models and vectorizer
vectorizer = joblib.load('bigram_tfidf_vectorizer.pkl')
scaler = joblib.load('scaler.pkl')
clf = joblib.load('logistic_regression_model.pkl')

stop_words = set(stopwords.words('english'))

def preprocess_email(email_text):
    """Cleans email text: lowercasing, removing punctuation, etc."""
    email_text = email_text.lower()
    email_text = email_text.translate(str.maketrans("", "", string.punctuation))
    return email_text

def extract_features(email_text):
    """Extract numerical features like word count, sentence count, etc."""
    words = email_text.split()
    sentences = sent_tokenize(email_text)
    Word_Count = len(words)  # Total words
    text_char = sum(len(word) for word in words)  # Total characters
    avg_word = text_char / Word_Count if Word_Count != 0 else 0
    stopwords = sum(1 for word in words if word in stop_words)  # Stopword count
    presence_of_digit = sum(char.isdigit() for char in email_text)  # Digit count
    sentence_num = len(sentences)

    return np.array([Word_Count, text_char, sentence_num, presence_of_digit, avg_word, stopwords]).reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict spam or ham."""
    data = request.get_json()
    email_text = data.get("text", "")

    # Preprocess and extract features
    processed_text = preprocess_email(email_text)
    numerical_features = extract_features(processed_text)
    normalized_features = scaler.transform(numerical_features)
    text_features = vectorizer.transform([processed_text])

    # Combine both sets of features
    combined_features = np.hstack((normalized_features, text_features.toarray()))

    # Predict using the model
    prediction = clf.predict(combined_features)[0]
    result = "Spam" if prediction == 1 else "Ham"

    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
