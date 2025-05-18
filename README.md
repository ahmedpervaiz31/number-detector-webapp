# Digit Recognition Web App

A minimal web app to classify hand-written digits (0-9) using a pre-trained Keras model.

## Features
- Upload a PNG or JPG image of a digit (0-9)
- The app preprocesses the image and predicts the digit using the provided model

## Folder Structure
```
/project-root
  /model
    softmax_mnist_model.keras
  /uploads
  /templates
    index.html
  /static
  app.py
  requirements.txt
  README.md
  venv/
```

## Setup (Windows, PowerShell)
1. **Create and activate virtual environment (if not already):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```powershell
   python app.py
   ```
4. **Open your browser and go to:**
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Usage
- Upload a clear image of a single digit (0-9) in PNG or JPG format.
- The app will display the predicted digit.

## Troubleshooting
- If you get TensorFlow or dependency errors, ensure your virtual environment is activated and all packages are installed.
- If the app can't find the model, make sure `softmax_mnist_model.keras` is in the `model/` folder.
- For any file upload issues, check that the `uploads/` folder exists and is writable.

## Notes
- This app is intentionally minimal. For production, add security, validation, and error handling. 