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

<img width="643" height="658" alt="Screenshot 2026-02-05 132222" src="https://github.com/user-attachments/assets/5959a08d-32f8-4202-8a16-b6090ea811b2" />

---
## System Architecture  

![system arc](https://github.com/user-attachments/assets/f4180e46-7e27-4eb4-b3a5-dbae3e0c4d4a)

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

#### Classification Report

<img width="532" height="274" alt="Screenshot 2026-02-05 132831" src="https://github.com/user-attachments/assets/5d4c1467-7a68-4ff3-8652-93e4c441476f" />

#### Confusion Matrix

<img width="654" height="588" alt="Screenshot 2026-02-05 132852" src="https://github.com/user-attachments/assets/68e7fa0f-ccd1-4653-9e96-c82074934c10" />

#### Per-Class Accuracy

<img width="284" height="106" alt="Screenshot 2026-02-05 132910" src="https://github.com/user-attachments/assets/32bae11d-ce23-4faa-b7ec-8fa60804121c" />

The trained InceptionV3 model demonstrated reliable performance across all four tumor classes.

---

##### Results  

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

--- 

## Key Model Implementation Code

```
# Load trained model
model = tf.keras.models.load_model("models/inceptionv3_best.keras")

# Predict tumor class
preds = model.predict(img_array)
pred_index = int(np.argmax(preds))
confidence = round(float(preds[0][pred_index]) * 100, 2)

# Generate Grad-CAM heatmap
heatmap = make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name="mixed10",
    pred_index=pred_index
)

```

---
## Output

### User Interface for Patient Data Entry and MRI Upload
<img width="635" height="584" alt="Screenshot 2026-02-05 135222" src="https://github.com/user-attachments/assets/52e2bf20-bb38-4e24-96ba-658a7b348f57" />

### Classification Result and Grad-CAM Visualization
<img width="620" height="813" alt="Screenshot 2026-02-05 135137" src="https://github.com/user-attachments/assets/c047d4dc-8a7c-4d00-a13c-d086fb74153c" />

### Generated Medical PDF Report
<img width="439" height="561" alt="Screenshot 2026-02-05 135154" src="https://github.com/user-attachments/assets/e8bd907e-8329-40cf-864f-eccd2cf26a7b" />

---
## Future Enhancements
 
🔹 Store patient history using MongoDB or Firebase

🔹 Cloud-based deployment for real-time inference

---
## References

[1] C. Szegedy et al., “Going Deeper with Convolutions,” IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

[2] R. R. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks,” ICCV, 2017.

[3] K. Zhou and X. Chen, “Explainable AI in Medical Image Analysis,” Medical Image Analysis, 2021.

[4] M. Havaei et al., “Brain Tumor Segmentation with Deep Neural Networks,” Medical Image Analysis, 2017.

---

