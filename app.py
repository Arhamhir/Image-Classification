import streamlit as st
import numpy as np
from PIL import Image
import joblib as jb 

model = jb.load('animal.pkl')

labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

st.set_page_config(page_title="Image Classifier", layout="centered")
st.title("Image Classification")
st.code(f"Accuracy: 69%")
st.markdown("Upload an image and the model will try to classify it")

st.markdown("### Available Labels")
st.code(" | ".join(labels))

upload_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

if upload_file is not None:
    img = Image.open(upload_file).convert('RGB')
    st.image(img, use_container_width=True)

    img_size = img.resize((32, 32))
    img_arr = np.array(img_size) / 255.0
    img_arr = img_arr.reshape(1, 32, 32, 3)

    prediction = model.predict(img_arr)
    confidence = np.max(prediction)
    label = np.argmax(prediction)

    st.markdown("---")
    st.success(f"Predicted: {labels[label]}")
    st.code(f"Confidence: {confidence*100:.2f}%")
