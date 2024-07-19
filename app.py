from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

VECTOR_PATH_1 = "D:/Flask/Demo Flask/FlaskAPi/models/vectorizer1.pkl"
CLASSIFIER_PATH_1 = "D:/Flask/Demo Flask/FlaskAPi/models/classifier1.pkl"
LABEL_BINARIZER_PATH_1 = "D:/Flask/Demo Flask/FlaskAPi/models/label_binarizer1.pkl"
VECTOR_PATH_4 = "D:/Flask/Demo Flask/FlaskAPi/models/vectorizer4.pkl"
CLASSIFIER_PATH_3 = "D:/Flask/Demo Flask/FlaskAPi/models/classifier3.pkl"
LABEL_BINARIZER_PATH_4 = "D:/Flask/Demo Flask/FlaskAPi/models/label_binarizer4.pkl"

vectorizer1 = None
classifier1 = None
label_binarizer1 = None
vectorizer4 = None
classifier3 = None
label_binarizer4 = None


def load_models():
    global vectorizer1, classifier1, label_binarizer1, vectorizer4, classifier3, label_binarizer4

    if os.path.exists(VECTOR_PATH_1):
        with open(VECTOR_PATH_1, 'rb') as f:
            vectorizer1 = pickle.load(f)

    if os.path.exists(CLASSIFIER_PATH_1):
        with open(CLASSIFIER_PATH_1, 'rb') as f:
            classifier1 = pickle.load(f)

    if os.path.exists(LABEL_BINARIZER_PATH_1):
        with open(LABEL_BINARIZER_PATH_1, 'rb') as f:
            label_binarizer1 = pickle.load(f)

    if os.path.exists(VECTOR_PATH_4):
        with open(VECTOR_PATH_4, 'rb') as f:
            vectorizer4 = pickle.load(f)

    if os.path.exists(CLASSIFIER_PATH_3):
        with open(CLASSIFIER_PATH_3, 'rb') as f:
            classifier3 = pickle.load(f)

    if os.path.exists(LABEL_BINARIZER_PATH_4):
        with open(LABEL_BINARIZER_PATH_4, 'rb') as f:
            label_binarizer4 = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/action_page', methods=['POST'])
def action_page():
    data = request.json
    email = data.get('email')

    if email is None or not email.strip():
        return jsonify({"error": "Please enter the question title."}), 400

    predicted_tags = predict_tags(email)
    return jsonify({"predicted_tags": predicted_tags})


def predict_tags(question_title):

    text_vector = vectorizer1.transform([question_title])
    prediction = classifier1.predict(text_vector)
    predicted_label = label_binarizer1.inverse_transform(prediction)
    return predicted_label


if __name__ == '__main__':
    load_models()
    app.run(debug=True, port=5000)
