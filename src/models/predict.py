import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
MODEL_PATH = "waste_model.h5"
model = load_model(MODEL_PATH)

# Class labels (change if your order is different)
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

IMG_SIZE = 224  # same size used during training


def preprocess_image(image_path):
    """
    Read and preprocess image for prediction
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_waste(image_path):
    """
    Predict waste category from image
    """
    processed_img = preprocess_image(image_path)
    predictions = model.predict(processed_img)
    class_index = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))

    return {
        "predicted_class": CLASS_NAMES[class_index],
        "confidence": round(confidence * 100, 2)
    }


# For testing directly
if __name__ == "__main__":
    test_image = "test.jpg"  # replace with your test image
    result = predict_waste(test_image)
    print("Prediction:", result)
