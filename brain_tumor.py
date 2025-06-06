from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load your trained CNN model
MODEL_PATH = 'model/brain_tumor_model.h5'
model = load_model(MODEL_PATH)

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction='No file uploaded.')

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction='No file selected.')

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)
        result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"

        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
