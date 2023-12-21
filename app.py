from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import pickle
import json
import numpy as np


app = Flask(__name__)
CORS(app)

# Load pre-trained model
model = load_model("my_model.h5")

# Load tokenizer using Pickle
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder using Pickle
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
    
# Load data from JSON file
with open('job_data.json') as json_file:
    data = json.load(json_file)
    
    
job_fields = data['job_fields']
X_text = []
            
for field in job_fields:
    field_name = field['field_name']
    for job in field['job_examples']:
        skills = ' '.join(job['skills'])  # Only use skills for prediction
        X_text.append(skills)

# Tokenize and pad sequences
X_encoded = tokenizer.texts_to_sequences(X_text)
X_padded = pad_sequences(X_encoded)

def predict_category_for_skills(model, tokenizer, skills):
    # Tokenize and pad the input skills
    skills_sequence = tokenizer.texts_to_sequences([skills])
    skills_padded = pad_sequences(skills_sequence, maxlen=X_padded.shape[1])

    # Get the predicted probabilities for each class
    predicted_probabilities = model.predict(skills_padded)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predicted_probabilities, axis=-1)[0]

    # Convert the predicted class index back to the original label
    predicted_class = label_encoder.classes_[predicted_class_index]

    return predicted_class
    
@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    skills_to_predict = data.get('skills', '')

    if not skills_to_predict:
        return jsonify({'error': 'No skills provided in the request.'}), 400

    try:
        # Make prediction
        predicted_category = predict_category_for_skills(model, tokenizer, skills_to_predict)
            
        # Return the prediction as JSON with a success status code
        return jsonify({
            "status": {
                "code": 200,
                "message": "Sukses Memprediksi Kategori Pekerjaan",
            },
            "data": {
                "selected_skills": skills_to_predict,
                "predictect_category": predicted_category
            }
                }),200
            
    except Exception as e:
        # Return an error message and status code 500 for internal server error
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Success Fetching the API",
        },
        "data": None
    }), 200
    

if __name__ == "__main__":
    app.run(debug=True)
    
    