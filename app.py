from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model('kidney_disease_model (4).h5')
class_names = ['Normal', 'Cyst', 'Tumor', 'Stone']

def is_valid_image(image_path):
    try:
        img = Image.open(image_path)
        
        if img.mode == 'RGB':
            img_array = np.array(img)
            threshold = 5  
            diff_rg = np.abs(img_array[..., 0] - img_array[..., 1])
            
            diff_gb = np.abs(img_array[..., 1] - img_array[..., 2])
            
            if np.max(diff_rg) > threshold or np.max(diff_gb) > threshold:
                return False
            
            img = img.convert('L')
        elif img.mode != 'L':
            
            img = img.convert('L')
        
       
        img_array = np.array(img)
        mean_intensity = np.mean(img_array)
        if mean_intensity > 250 or mean_intensity < 5:
            return False
        
        return True
        
    except Exception as e:
        return False


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype=np.float32) / 255.0  
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({"error": "No image uploaded."})
        image_path = 'uploaded_image.png'
        file.save(image_path)

        if not is_valid_image(image_path):
            os.remove(image_path)
            return jsonify({"error": "This is not a kidney radiology image."})
        
        processed_image = preprocess_image(image_path)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Prediction: {class_names[predicted_class]} (Confidence: {confidence * 100:.2f}%)")
        plt.show()

        os.remove(image_path)

        result = {
            "prediction": class_names[predicted_class],
            "confidence": f"{confidence * 100:.2f}%"
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
