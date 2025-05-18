import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Updated for Railway deployment
UPLOAD_FOLDER = '/tmp' if os.environ.get('RAILWAY') else 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'model/softmax_mnist_model.tf.keras'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

# Load model with custom_objects to handle version compatibility
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to loading with custom_objects
    model = tf.keras.models.load_model(MODEL_PATH, 
                                     custom_objects={'Sequential': tf.keras.Sequential},
                                     compile=False)
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # grayscale
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255.0  # normalize
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img_array = preprocess_image(filepath)
            prediction = model.predict(img_array)
            digit = np.argmax(prediction)
            os.remove(filepath)
            return render_template('index.html', digit=digit)
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)
    return render_template('index.html', digit=None)

# Add back the local development server
if __name__ == '__main__':
    if not os.environ.get('RAILWAY'):
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True) 