from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# Load models with error handling
models = {}

# Load the models
try:
    models['breast_cancer'] = load_model('models/breast_cancer_model.keras')
    models['pneumonia'] = load_model('models/pneu_cnn_model.keras')
    models['tb'] = load_model('models/tb.keras')  # Correct reference for TB model
    models['ecg'] = load_model('models/ecg.keras')  # Correct reference for ECG model
    print("Models loaded successfully:", models.keys())
except Exception as e:
    print(f"Error loading model: {e}")

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Pneumonia Detection Page
@app.route('/pnemonia')
def pneumonia_page():
    return render_template('pnemonia.html')

# Breast Cancer Detection Page
@app.route('/cancer')
def breast_cancer_page():
    return render_template('cancer.html')

# TB Detection Page
@app.route('/tb')
def tb_page():
    return render_template('tb.html')

# ECG Detection Page
@app.route('/ecg')
def ecg_page():
    return render_template('ecg.html')

# About us page
@app.route('/about')
def About_Us_page():
    return render_template('about.html')
# symptounms page
@app.route('/symptounms')
def symptounms_page():
    return render_template('symptounms.html')


# Route to handle Pneumonia prediction
@app.route('/predict_symptoms', methods=['POST'])
def predict_symptoms():
    symptoms = request.form.get('symptoms')
    if not symptoms or symptoms.strip() == "":
        return "No symptoms provided", 400

    # Dummy prediction logic based on keywords in the symptom text.
    # Replace this logic with your actual model or decision system.
    symptoms_lower = symptoms.lower()
    if "cough" in symptoms_lower or "weight loss" in symptoms_lower or "night sweats" in symptoms_lower:
        prediction = "Tuberculosis"
    elif "lump" in symptoms_lower or "nipple discharge" in symptoms_lower:
        prediction = "Breast Cancer"
    elif "chest pain" in symptoms_lower or "shortness of breath" in symptoms_lower:
        prediction = "Pneumonia"
    elif "irregular heartbeat" in symptoms_lower or "palpitations" in symptoms_lower:
        prediction = "ECG (Abnormal)"
    else:
        prediction = "Unknown - please consult a doctor"

    # For demonstration, we set a dummy accuracy value.
    accuracy = 100

    return render_template('result.html', disease=prediction, result="Predicted", accuracy=accuracy)

# Other routes for image-based predictions (if needed)
@app.route('/pnemonia', methods=['POST'])
def predict_pneumonia():
    if 'pneumonia_image' in request.files:
        img = request.files['pneumonia_image']
        img_path = os.path.join('static', 'uploads', img.filename)
        img.save(img_path)

        image = load_img(img_path, target_size=(150, 150))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0

        prediction = models['pneumonia'].predict(image)
        result = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        accuracy = prediction[0][0] if result == 'Positive' else 1 - prediction[0][0]

        return render_template('result.html', disease='Pneumonia', result=result, accuracy=round(accuracy * 100, 2))

    return "No image provided", 400

# Route to handle Breast Cancer prediction
@app.route('/cancer', methods=['POST'])
def predict_breast_cancer():
    if 'breast_cancer_image' in request.files:
        img = request.files['breast_cancer_image']
        img_path = os.path.join('static', 'uploads', img.filename)
        img.save(img_path)

        image = load_img(img_path, target_size=(150, 150))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0

        prediction = models['breast_cancer'].predict(image)
        result = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        accuracy = prediction[0][0] if result == 'Positive' else 1 - prediction[0][0]

        return render_template('result.html', disease='Breast Cancer', result=result, accuracy=round(accuracy * 100, 2))

    return "No image provided", 400

# Route to handle TB prediction
@app.route('/tb', methods=['POST'])
def predict_tb():
    if 'tb_image' in request.files:
        img = request.files['tb_image']
        img_path = os.path.join('static', 'uploads', img.filename)
        img.save(img_path)

        # Resize the image to a smaller size like 150x150 if needed (verify this size with the model)
        image = load_img(img_path, target_size=(150, 150))  # Adjust size as needed
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize the image

        # Check the shape of the image for debugging
        print(f"Image shape: {image.shape}")

        # Predict using the loaded model
        prediction = models['tb'].predict(image)

        result = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        accuracy = prediction[0][0] if result == 'Positive' else 1 - prediction[0][0]

        return render_template('result.html', disease='Tuberculosis', result=result, accuracy=round(accuracy * 100, 2))

    return "No image provided", 400

# Route to handle ECG prediction
@app.route('/ecg', methods=['POST'])
def predict_ecg():
    if 'ecg_image' in request.files:
        img = request.files['ecg_image']
        img_path = os.path.join('static', 'uploads', img.filename)
        img.save(img_path)

        image = load_img(img_path, target_size=(150, 150))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0

        prediction = models['ecg'].predict(image)
        result = 'Abnormal' if prediction[0][0] > 0.5 else 'Normal'
        accuracy = prediction[0][0] if result == 'Abnormal' else 1 - prediction[0][0]

        return render_template('result.html', disease='ECG', result=result, accuracy=round(accuracy * 100, 2))

    return "No image provided", 400

if __name__ == "__main__":
    app.run(debug=True)
