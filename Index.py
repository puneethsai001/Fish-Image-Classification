import streamlit as st
from PIL import Image
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Fish Image Classification", layout="wide")
st.title("Multiclass Fish Image Classification")
st.text("By Puneeth Sai")

input_file = st.file_uploader("Upload Your Image (JPG only)", type="jpg", accept_multiple_files=False)

if input_file is not None and st.button("Classify"):

    image = Image.open(input_file)
    st.image(image, caption="Uploaded image", width=300)

    # Load model fresh each time
    model = load_model("MobileNet/mobilenet_finetuned_model.h5")

    # Load class labels
    with open("class_labels.json", "r") as f:
        labels = json.load(f)

    if isinstance(labels, dict):
        labels = [labels[str(i)] for i in range(len(labels))]

    # Preprocess the image
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Predict
    preds = model.predict(arr)
    probs = preds[0]
    top_k = 3
    top_idx = probs.argsort()[-top_k:][::-1]

    st.subheader("Prediction")
    st.write(f"**Top-1:** {labels[top_idx[0]]} — {probs[top_idx[0]]:.2%}")

    st.subheader("Top-3 probabilities")
    for i in top_idx:
        st.write(f"{labels[i]}: {probs[i]:.2%}")
