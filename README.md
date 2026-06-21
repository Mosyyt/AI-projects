<div align="center">

# AI Projects

**A collection of machine learning and AI projects covering classification, image recognition, and web interfaces.**

![Language](https://img.shields.io/badge/Language-Python%203.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ML](https://img.shields.io/badge/ML-TensorFlow%20%2F%20Keras%20%2F%20Scikit--learn-orange?style=for-the-badge)

</div>

---

## What is this?

Projects from my journey into machine learning and AI. The topics range from simple regression and classification notebooks to a CNN-based image classifier and a Streamlit app for plant disease detection. Each project is self-contained in its own folder.

---

## Projects

### StudentPerformancePrediction-ML

Predicts student grades using a dataset of behavioral factors — attendance, hands raised, study hours. Multiple classifiers are compared and results are visualized with confusion matrices and graphs.

```bash
cd StudentPerformancePrediction-ML
python Project.py
```

**Stack:** Python, Scikit-learn, Pandas, Matplotlib

---

### AI Object Detection

Real-time object detection in the browser via webcam. Uses the pretrained MobileNet model through ml5.js — no backend, no install, runs entirely client-side.

```bash
# Open index.html directly in your browser
cd ai-object-detection
```

**Stack:** HTML, CSS, JavaScript, ml5.js

---

### Leaf Diseases Detection

Identifies 33 plant diseases from leaf photos using a CNN trained with TensorFlow and Keras. Served as a Streamlit web app. Supports Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, and Tomato.

```bash
cd leaf-diseases-detect
pip install -r requirements.txt
streamlit run main.py
```

**Stack:** Python, TensorFlow, Keras, Streamlit, OpenCV

---

### Machine — ML Notebooks

Jupyter Notebooks covering the three core ML areas:

| Notebook | Topic |
|---|---|
| `Regrssion_learning.ipynb` | Linear and polynomial regression |
| `Classification_learning.ipynb` | KNN, SVM and other classifiers |
| `unsupervised_ML.ipynb` | Clustering and PCA |

```bash
cd machine && jupyter notebook
```

---

### Machine2 — Cat vs. Dog Classifier

Flask web app that classifies uploaded images as cat or dog using a Keras CNN. Includes OCR experiments and multiple datasets.

```bash
cd machine2
pip install flask keras tensorflow
python app.py
```

---

### Machineproject — Final ML Project

Complete project with regression, classification, and a Flask interface. Includes documentation.

```bash
cd machineproject && python app.py
```

---

## General Setup

```bash
git clone https://github.com/Mosyyt/AI-projects.git
cd AI-projects/<project-folder>
pip install -r requirements.txt
```
