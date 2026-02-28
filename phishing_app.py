from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Extract features from URL
def extract_features(url):
    return pd.Series({
        'length': len(url),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_digits': sum(c.isdigit() for c in url),
        'has_https': int('https' in url.lower()),
        'suspicious_words': int(any(word in url.lower() for word in ['login', 'verify', 'account', 'secure', 'update']))
    })

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    features = extract_features(url).to_frame().T
    result = model.predict(features)[0]
    prediction = "Phishing" if result == 1 else "Begin"
    return render_template('index.html', prediction_text=f"Prediction: {prediction}")

if __name__ == '__main__':
    app.run(debug=True)
