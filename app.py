
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="OCT Disease Classifier", layout="centered")

st.title("🧠 OCT Scan Retinal Disease Classifier")
st.markdown(
    "Upload an OCT image to detect CNV, DME, or DRUSEN. "
    "If none of them is confidently detected, the system will classify it as NORMAL."
)

# Load all three models only once
@st.cache_resource
def load_models():
    cnv_model = tf.keras.models.load_model("cnv_model.keras")
    dme_model = tf.keras.models.load_model("dme_model.keras")
    drusen_model = tf.keras.models.load_model("drusen_model.keras")
    return cnv_model, dme_model, drusen_model

cnv_model, dme_model, drusen_model = load_models()

# Function to preprocess uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    if img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel if present
    img = np.expand_dims(img, axis=0)
    return img

# Upload image
uploaded_file = st.file_uploader("Upload OCT Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded OCT Image", use_column_width=True)

    # Preprocess and predict
    img_array = preprocess_image(image)
    cnv_prob = cnv_model.predict(img_array)[0][0]
    dme_prob = dme_model.predict(img_array)[0][0]
    drusen_prob = drusen_model.predict(img_array)[0][0]

    results = {
        "CNV": cnv_prob,
        "DME": dme_prob,
        "DRUSEN": drusen_prob
    }

    max_class = max(results, key=results.get)
    final_prediction = max_class if results[max_class] >= 0.5 else "NORMAL"

    st.markdown("### 🔬 Prediction Probabilities:")
    for k, v in results.items():
        st.write(f"{k}: {v*100:.2f}%")

    st.markdown(f"### ✅ Final Diagnosis: **{final_prediction}**")
