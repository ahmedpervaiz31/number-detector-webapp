import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ---------------------------
# 1. Data Preparation
# ---------------------------
train_df = pd.read_csv("train.csv")

y = train_df.iloc[:, 0].values  # Labels (digits 0-9)
X = train_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)  # Reshape to (num_samples, 28, 28, 1)

X = X / 255.0  # Normalize pixel values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_cat = to_categorical(y_train, num_classes=10)
y_valid_cat = to_categorical(y_valid, num_classes=10)

X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)
X_valid = np.nan_to_num(X_valid)
y_valid = np.nan_to_num(y_valid)

# ---------------------------
# 2. Model: CNN (LeNet-5)
# ---------------------------

model = tf.keras.Sequential([
    # C1: Convolutional Layer
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # C3: Convolutional Layer
    tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten before Fully Connected layers
    tf.keras.layers.Flatten(),

    # FC Layer with 120 units
    tf.keras.layers.Dense(120, activation='relu'),

    # FC Layer with 84 units
    tf.keras.layers.Dense(84, activation='relu'),

    # Output Layer: 10 neurons for 10 classes
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_cat,
                    epochs=10,
                    batch_size=64,
                    validation_data=(X_valid, y_valid_cat))

# Optional: Evaluate and print accuracy
val_loss, val_acc = model.evaluate(X_valid, y_valid_cat)
print(f"Validation Accuracy: {val_acc:.4f}")

# Save the trained model for use in app.py
model.save('model/softmax_mnist_model.tf.keras')
print("Model weights saved to 'model/softmax_mnist_model.tf.keras'")

