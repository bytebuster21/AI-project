import os
import numpy as np
from flask import Flask, request, send_from_directory
from flask import send_file
from flask import render_template_string
from flask import redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the model
MODEL_PATH = 'resnet50_synapsescan.h5'
model = load_model(MODEL_PATH)

# Class labels
class_labels = ['Class_A', 'Class_B', 'Class_C']

# Read HTML and CSS content
def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f:
        return f.read()

# Home Page
@app.route('/')
def home():
    with open("index.html", "r") as f:
        html = f.read()
    return render_template_string(html, prediction=None)

# About Page
@app.route('/about')
def about():
    return send_file('about.html')
    
# Contact Page
@app.route('/contact')
def contact():
    return send_file('contact.html')

# Serve CSS
@app.route('/style.css')
def serve_css():
    return send_from_directory(os.path.dirname(__file__), 'style.css')

# Handle image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template_string(read_file("index.html"), prediction="No image uploaded.")

    file = request.files['file']
    if file.filename == '':
        return render_template_string(read_file("index.html"), prediction="No file selected.")

    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Preprocess and predict
    img = image.load_img(filepath, target_size=(299, 299))  # Match your model input
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_class = class_labels[predicted_index]

    result = f"Predicted: {predicted_class}<br>Confidence: {confidence:.2f}%"

    # Render HTML with prediction
    with open("index.html", "r") as f:
        html = f.read()
    return render_template_string(html, prediction=result)

# Run app
if __name__ == '__main__':
    app.run(debug=True)
