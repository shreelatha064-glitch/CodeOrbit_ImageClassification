import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pretrained MobileNetV2 model
print("Loading MobileNetV2 model...")

model = MobileNetV2(weights="imagenet")

print("Model loaded successfully!\n")

# Ask the user for an image path
image_path = input("Enter the path of the image: ")

# Load and resize the image
img = image.load_img(image_path, target_size=(224, 224))

# Convert image to NumPy array
img_array = image.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Preprocess the image
img_array = preprocess_input(img_array)

# Make prediction
predictions = model.predict(img_array)

# Get the top 5 predictions
results = decode_predictions(predictions, top=5)[0]

print("\nTop 5 Predictions:")
print("-------------------")

for i, (_, label, probability) in enumerate(results, start=1):
    print(f"{i}. {label}: {probability * 100:.2f}%")