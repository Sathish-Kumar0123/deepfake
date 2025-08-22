import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "deepfake_model.h5")

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "Real" if prediction < 0.5 else "Fake"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]

    # If relative path given, join with BASE_DIR
    if not os.path.isabs(img_path):
        img_path = os.path.join(BASE_DIR, img_path)

    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        sys.exit(1)

    result = predict_image(img_path)
    print(f"✅ Prediction: {result}")
