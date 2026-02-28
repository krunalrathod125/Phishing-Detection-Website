import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Dummy sample data
data = {
    'URL': ['paypal-login.com', 'google.com', 'secure-bank-login.net', 'microsoft.com'],
    'Label': [1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Convert Label
df['Label'] = df['Label'].map({'benign': 0, 'phishing': 1}) if df['Label'].dtype == 'object' else df['Label']

# Feature extraction
def extract_features(url):
    return pd.Series({
        'length': len(url),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_digits': sum(c.isdigit() for c in url),
        'has_https': int('https' in url.lower()),
        'suspicious_words': int(any(word in url.lower() for word in ['login', 'verify', 'update', 'account', 'secure']))
    })

X = df['URL'].apply(extract_features)
y = df['Label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ✅ Save model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as model.pkl ✅")
