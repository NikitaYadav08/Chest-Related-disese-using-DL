# convert_model.py
import os
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Sample data - replace with actual data
X_train = np.random.rand(100, 150, 150, 3)  # 100 samples, 150x150 images
y_train = np.random.randint(0, 2, 100)  # Binary classification

# Define a simple model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)

# Save the model in .h5 format (for testing)
h5_path = "models/ecg.h5"
os.makedirs("models", exist_ok=True)
model.save(h5_path)

# Function to convert .h5 to .keras format
def convert_h5_to_keras(h5_path, keras_path):
    try:
        # Load model without optimizer states (compile=False)
        model = load_model(h5_path, compile=False)
        
        # Manually compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Save the model in .keras format
        model.save(keras_path)
        print(f"Converted {h5_path} to {keras_path}")
    except Exception as e:
        print(f"Error converting {h5_path} to {keras_path}: {e}")

# Call the conversion function for your models

convert_h5_to_keras("models/tb.h5", "models/tb.keras")
convert_h5_to_keras("models/breast_cancer.h5", "models/breast_cancer.keras")
convert_h5_to_keras("models/pneumonia.h5", "models/pneumonia.keras")


