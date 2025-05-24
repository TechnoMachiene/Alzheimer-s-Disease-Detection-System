# 🧠 Alzheimer's Disease Detection System

A deep learning and machine learning-based system for the early detection and classification of Alzheimer’s Disease using both **MRI imaging** and **clinical data**. The system integrates an ensemble model to enhance diagnostic accuracy across four categories: **AD, CN, EMCI, and LMCI**.

---

## 📁 Project Structure

alzheimer-api/
│
├── models/ # Pre-trained models
│ ├── best_model_2.pth # PyTorch CNN model
│ ├── T_L_1.keras # Keras EfficientNetB4 model
│ ├── ensemble_pipeline_1.pkl # Combined ensemble model (CNN + EfficientNet)
│ └── xgboost_target1_group.pkl # XGBoost model for clinical data
│
├── static/
│ ├── media/ # Uploaded images
│ └── style.css # Stylesheet for frontend
│
├── templates/
│ ├── clinical.html # Page for clinical data prediction
│ ├── home.html # Landing page
│ └── mri.html # Page for MRI image prediction
│
├── uploads/ # Temporary uploaded files
│
├── app.py # Flask app entry point
│
├── Jupyter notebooks/ # Model training and evaluation notebooks
│ ├── AD-CNN.ipynb
│ ├── EfficientNet-B4.ipynb
│ ├── Ensemble Model.ipynb
│ └── XGBoost (Clinical Data).ipynb
│
└── README.md


---

## 🚀 Features

- 🧠 MRI image-based classification using custom CNN and EfficientNetB4
- 📊 Clinical data-based prediction using XGBoost
- 🔗 Ensemble learning combining deep learning and classical ML models
- 🌐 Web-based user interface using Flask
- 📂 Upload system for MRI images and patient data

---

## 🛠️ Technologies Used

- **Python 3.10**
- **Flask**
- **PyTorch**
- **TensorFlow/Keras**
- **XGBoost**
- **Scikit-learn**
- **HTML/CSS (Jinja2 templates)**

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/alzheimer-api.git
cd alzheimer-api

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask application
python app.py

