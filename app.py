from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

client = OpenAI(api_key="your-openai-key")

app = Flask(__name__)
CORS(app)

# Set your OpenAI API key

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_input}]
    )
    return jsonify({"response": response.choices[0].message.content})

if __name__ == '__main__':
    app.run(debug=True)

# Model Training for Bug Prediction
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('bug_data.csv')

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2)

# Create a pipeline with a TfidfVectorizer and a RandomForestClassifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier())
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Evaluate the pipeline on the test data
accuracy = pipeline.score(X_test, y_test)
print(f'Model Accuracy: {accuracy}')


# Integrate bug prediction model with Flask
@app.route('/predict-bug', methods=['POST'])
def predict_bug():
    task_description = request.json.get('task')
    prediction = pipeline.predict([task_description])[0]
    return jsonify({'prediction': prediction})