# MNIST Digit Recognition Web App

A web application that recognizes handwritten digits using a CNN model trained on the MNIST dataset.

## Features
- Upload handwritten digit images
- Real-time digit recognition
- Built with Flask and TensorFlow
- Deployed on Vercel

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Run the application:
```bash
python app.py
```

## Technologies Used
- Python
- Flask
- TensorFlow
- HTML/CSS
- Vercel (Deployment)

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

## Usage
- Upload a clear image of a single digit (0-9) in PNG or JPG format.
- The app will display the predicted digit.

## Troubleshooting
- If you get TensorFlow or dependency errors, ensure your virtual environment is activated and all packages are installed.
- If the app can't find the model, make sure `softmax_mnist_model.keras` is in the `model/` folder.
- For any file upload issues, check that the `uploads/` folder exists and is writable.

## Notes
- This app is intentionally minimal. For production, add security, validation, and error handling. 