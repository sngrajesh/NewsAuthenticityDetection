import re
import pickle
import nltk
from nltk.corpus import stopwords
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# Load the model and vectorizer
with open('models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase and remove stopwords
    words = text.lower().split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)


# For the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if request.form['news_text'] == '':
            return 'Please provide a news text.'
        news_text = request.form['news_text']
        preprocessed_text = preprocess_text(news_text)
        vectorized_text = vectorizer.transform([preprocessed_text])
        prediction = model.predict(vectorized_text)[0]        
        return str(prediction)


# For the web ui
@app.route('/', methods=['GET', 'POST'])
def home(): 
    prediction = None
    if request.method == 'POST':
        if request.form['news_text'] == '':
            return render_template('index.html', prediction=prediction)
        news_text = request.form['news_text']
        preprocessed_text = preprocess_text(news_text)
        vectorized_text = vectorizer.transform([preprocessed_text])
        prediction = model.predict(vectorized_text)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)