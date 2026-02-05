from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
from fpdf import FPDF
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "models/inceptionv3_best.keras"
UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"
REPORT_FOLDER = "static/reports"

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "mixed10"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# GRAD-CAM (FINAL SAFE VERSION)
# -----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predictions = tf.reshape(predictions, (-1,))
        class_channel = predictions[pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def apply_gradcam(img_path, heatmap, out_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    final = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(out_path, final)

# -----------------------------
# PDF REPORT
# -----------------------------
def generate_pdf(data, gradcam_path, pdf_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Brain Tumor Analysis Report", ln=True, align="C")
    pdf.ln(5)

    for k, v in data.items():
        pdf.cell(200, 8, f"{k}: {v}", ln=True)

    pdf.ln(5)
    pdf.image(gradcam_path, x=30, w=150)
    pdf.output(pdf_path)

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        patient = request.form
        file = request.files["mri"]

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        patient_id = f"BT-{timestamp}"

        img_filename = f"{timestamp}.jpg"
        img_path = os.path.join(UPLOAD_FOLDER, img_filename)
        file.save(img_path)

        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        preds = model.predict(img_array)
        pred_index = int(np.argmax(preds))
        confidence = round(float(preds[0][pred_index]) * 100, 2)

        heatmap = make_gradcam_heatmap(
            img_array, model, LAST_CONV_LAYER, pred_index
        )

        gradcam_filename = img_filename
        gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_filename)
        apply_gradcam(img_path, heatmap, gradcam_path)

        report_data = {
            "Patient ID": patient_id,
            "Patient Name": patient["name"],
            "Age": patient["age"],
            "Gender": patient["gender"],
            "Prediction": CLASSES[pred_index],
            "Confidence (%)": confidence,
            "Scan Date": datetime.now().strftime("%d-%m-%Y")
        }

        pdf_filename = img_filename.replace(".jpg", ".pdf")
        pdf_path = os.path.join(REPORT_FOLDER, pdf_filename)
        generate_pdf(report_data, gradcam_path, pdf_path)

        return render_template(
            "result.html",
            data=report_data,
            gradcam=gradcam_filename,
            pdf=pdf_filename
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
