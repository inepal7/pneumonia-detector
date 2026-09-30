"""
Runs the trained pneumonia detector on a single X-ray image.

Usage: python src/predict.py path/to/xray.jpg
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parent.parent / "pneumonia_model.keras"
IMG_SIZE = (224, 224)


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/xray.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not MODEL_PATH.exists():
        print(f"No trained model found at {MODEL_PATH}. Run train_model.py first.")
        sys.exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)

    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    confidence = prob if prob > 0.5 else 1 - prob

    print(f"Prediction: {label} (confidence: {confidence:.1%})")
    print("\nNote: this is a learning project, not a diagnostic tool.")


if __name__ == "__main__":
    main()
