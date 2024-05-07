import azure.functions as func
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load JSON data and preprocess patterns
with open('intents.json', 'r') as file:
    intents = json.load(file)

labels = []
patterns = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        labels.append(intent['tag'])
        patterns.append(pattern.lower())  # Convert patterns to lowercase for case insensitivity

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)

def get_most_similar_intent(user_input, vectorizer, X, labels):
    user_vector = vectorizer.transform([user_input.lower()])  # Convert user input to lowercase
    similarity_scores = cosine_similarity(user_vector, X)
    most_similar_index = np.argmax(similarity_scores)
    return labels[most_similar_index]

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()
        user_input = req_body['message']
        intent = get_most_similar_intent(user_input, vectorizer, X, labels)
        for intent_data in intents['intents']:
            if intent_data['tag'] == intent:
                responses = intent_data['responses']
                return func.HttpResponse(json.dumps({"Chatbot": np.random.choice(responses)}), mimetype="application/json")
    except ValueError:
        pass  # Handle any value errors here

    return func.HttpResponse(
        "Invalid request",
        status_code=400
    )
