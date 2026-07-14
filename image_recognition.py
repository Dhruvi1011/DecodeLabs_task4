import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image
import os

print("=" * 50)
print("      IMAGE RECOGNITION USING MOBILENETV2")
print("=" * 50)

# Check image exists
image_path = r"D:\Internships_2ndYear\DecodeLabs_AI\sample_pic.jpeg" # Replace with your image path

if not os.path.exists(image_path):
    print(f"Error: '{image_path}' not found.")
    print("Place an image named 'sample_pic.jpeg' in the project folder.")
    exit()

print("\nLoading pre-trained model...")
model = MobileNetV2(weights="imagenet")

print("Model loaded successfully!")

# Load image
print("\nLoading image...")
img = image.load_img(image_path, target_size=(224, 224))

# Convert image to array
img_array = image.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Preprocess image
img_array = preprocess_input(img_array)

print("Analyzing image...")

# Prediction
predictions = model.predict(img_array)

# Decode predictions
results = decode_predictions(predictions, top=3)[0]

print("\nTop 3 Predictions:")
print("-" * 50)

for i, (_, label, confidence) in enumerate(results, start=1):
    print(f"{i}. {label} : {confidence * 100:.2f}%")

print("\nImage Recognition Completed Successfully!")