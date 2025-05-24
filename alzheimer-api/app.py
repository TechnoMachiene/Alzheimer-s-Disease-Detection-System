from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from torchvision import transforms
import clip
from PIL import Image
import joblib
import os

app = Flask(
    __name__,
    template_folder="templates",  # ← move all your .html files here
    static_folder="static"        # ← keep your style.css (and any images) here
)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load Models (Move this to app initialization)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP Model
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Load XGBoost Model
xgboost_model = joblib.load("models/xgboost_target1_group.pkl")

# Custom Keras Layer
class AttentionModule(Layer):
    def __init__(self, **kwargs):
        super(AttentionModule, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                               initializer="glorot_uniform",
                               trainable=True)
        super(AttentionModule, self).build(input_shape)
    def call(self, x):
        attention = tf.matmul(x, self.W)
        attention = tf.nn.softmax(attention, axis=-1)
        return tf.multiply(x, attention)

# Load Keras Model
keras_model = load_model("models/T_L_1.keras", custom_objects={"AttentionModule": AttentionModule})

# Define PyTorch Model
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 28 * 28, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 4)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load PyTorch Model
pytorch_model = CNNModel().to(device)
pytorch_model.load_state_dict(torch.load("models/best_model_2.pth", map_location=device))
pytorch_model.eval()

# Preprocessing
pytorch_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    image = image.convert("RGB")
    pytorch_input = pytorch_transform(image).unsqueeze(0).to(device)
    keras_input = np.array(image.resize((224, 224)), dtype=np.float32)
    keras_input = (keras_input - np.mean(keras_input)) / np.std(keras_input)
    keras_input = np.expand_dims(keras_input, axis=0)
    return pytorch_input, keras_input

def is_brain_mri(image):
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    text_inputs = clip.tokenize([
        "a medical brain MRI scan in grayscale showing brain anatomy",
        "a color photograph of everyday objects",
        "an X-ray image of bones",
        "a microscopic cell image"
    ]).to(device)
    with torch.no_grad():
        logits_per_image, _ = clip_model(image_input, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    brain_mri_prob = probs[0][0]
    return brain_mri_prob > 0.35 and brain_mri_prob > np.max(probs[0][1:])

def preprocess_xgboost(age, gender, educ, ses, mmse, etiv, nwbv, asf):
    feature_means = np.array([77.0134, 14.597855, 2.460452, 27.342318, 
                             1488.128686, 0.729568, 1.195461])
    feature_stds = np.array([7.640957, 2.876339, 1.134005, 3.683244, 
                            176.139286, 0.037135, 0.138092])
    gender_num = 1 if gender == "M" else 0
    age_group = 1 if age > 75 else 0
    nwbv_etiv = nwbv / etiv
    input_data = np.array([[gender_num, age, educ, ses, mmse, etiv, nwbv, asf, age_group, nwbv_etiv]])
    input_data[:, [1,2,3,4,5,6,7]] = (input_data[:, [1,2,3,4,5,6,7]] - feature_means) / feature_stds
    return input_data

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/mri')
def mri():
    return render_template('mri.html')

@app.route('/clinical')
def clinical():
    return render_template('clinical.html')


@app.route('/predict/mri', methods=['POST'])
def predict_mri():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'})
    
    try:
        image = Image.open(file.stream).convert('RGB')
        if not is_brain_mri(image):
            return jsonify({'error': 'Uploaded image is not a valid brain MRI scan'})
        
        pytorch_input, keras_input = preprocess_image(image)
        
        with torch.no_grad():
            pytorch_logits = pytorch_model(pytorch_input)
            pytorch_probs = F.softmax(pytorch_logits, dim=1).cpu().numpy()
        
        keras_probs = keras_model.predict(keras_input)
        ensemble_probs = (0.4 * pytorch_probs) + (0.6 * keras_probs)
        predicted_class = np.argmax(ensemble_probs, axis=1)[0]
        
        class_labels = {
            0: 'Moderate Dementia (AD)',
            1: 'Non Dementia (CN)',
            2: 'Very Mild Dementia (EMCI)',
            3: 'Mild Dementia (LMCI)'
        }
        
        return jsonify({
            'prediction': class_labels[predicted_class],
            'confidence': float(np.max(ensemble_probs))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/clinical', methods=['POST'])
def predict_clinical():
    data = request.json
    required_fields = ['age', 'gender', 'educ', 'ses', 'mmse', 'etiv', 'nwbv', 'asf']
    
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        xgboost_input = preprocess_xgboost(
            float(data['age']),
            data['gender'],
            float(data['educ']),
            float(data['ses']),
            float(data['mmse']),
            float(data['etiv']),
            float(data['nwbv']),
            float(data['asf'])
        )
        
        xgboost_pred = xgboost_model.predict(xgboost_input)[0]
        group_labels = {0: "Non-Demented", 1: "Demented"}
        
        return jsonify({
            'prediction': group_labels[xgboost_pred]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)