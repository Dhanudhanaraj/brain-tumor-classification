# TEAM.NO.49: An Intelligent System for Brain Tumor Classification Using MRI Scans

## About  

This project focuses on detecting and classifying **Brain Tumors using MRI scans** through a deep learning–based intelligent system. The system classifies brain MRI images into the following categories:

1. Glioma  
2. Meningioma  
3. Pituitary  
4. No Tumor  

The solution employs a **pretrained InceptionV3 model using transfer learning**, where the convolutional layers are frozen and a custom classification head is trained on the MRI dataset. To improve interpretability, **Grad-CAM** is used to highlight tumor-affected regions in the MRI images.

A **Flask-based web application** allows users to upload MRI images, enter patient details, receive predictions with confidence scores, visualize Grad-CAM results, and download a **medical-style PDF report**.

---

## Features  

📤 Upload brain MRI image  

🤖 Real-time prediction using deep learning  

📊 Confidence score for predicted class  

📝 Enter patient details before prediction  

🔥 Grad-CAM–based tumor localization  

📄 Downloadable PDF medical report  

---

## Development Requirements  


## System Architecture  
2  

---

## Methodology  

### 1. Data Preprocessing  

i) The MRI images from the Brain Tumor MRI Dataset were organized into four classes: glioma, meningioma, pituitary, and no tumor.  

ii) All images were resized to **224 × 224 pixels**, normalized, and converted into RGB format suitable for CNN processing.  

iii) Basic preprocessing ensured consistent image quality and improved model stability during training.  

---

### 2. Model Training  

i) A deep learning model was used for feature extraction and classification:

1. InceptionV3 (Pretrained on ImageNet – Transfer Learning)

ii) The pretrained InceptionV3 convolutional layers were frozen, and a custom classification head consisting of Global Average Pooling, Dense, and Dropout layers was added.

iii) The model was trained in **Google Colab using GPU acceleration** with Adam optimizer and categorical cross-entropy loss.

The final trained model was saved as:  `inceptionv3_best.keras`

---

### 3. Model Evaluation  

Evaluation metrics included: accuracy, precision, recall, F1-score, and confusion matrix.

The trained InceptionV3 model demonstrated reliable performance across all four tumor classes.

---

#### Results  

The final deployed model achieved strong classification accuracy on the brain MRI dataset, effectively distinguishing between glioma, meningioma, pituitary tumors, and normal cases.

This system supports automated brain tumor classification and provides visual explanations through Grad-CAM, which may assist in clinical research and academic analysis.

---

### 4. Setup Instructions  

#### Run the Flask Web App:  

```
pip install -r requirements.txt
python app.py
```

#### Access Web Interface:
```
http://127.0.0.1:5000
```

## Key Model Implementation Code
```
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

base_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
output = Dense(4, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
```
## Output

### Web-page asking for patient details and MRI image upload

### Web-page result page


## Future Enhancements

🔹 Fine-tuning InceptionV3 layers for improved accuracy

🔹 Ensemble learning with EfficientNet or DenseNet 
 
🔹 Store patient history using MongoDB or Firebase

🔹 Cloud-based deployment for real-time inference

## References

[1] C. Szegedy et al., “Going Deeper with Convolutions,” IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

[2] R. R. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks,” ICCV, 2017.

[3] K. Zhou and X. Chen, “Explainable AI in Medical Image Analysis,” Medical Image Analysis, 2021.
