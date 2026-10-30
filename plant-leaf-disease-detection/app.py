"""Streamlit web interface.  Run:  streamlit run app.py"""
import numpy as np
import streamlit as st
from PIL import Image

import config
from src.preprocessing import (
    denoise, enhance_contrast, segment_leaf, diseased_area_ratio, diseased_overlay,
)
from src.predict import predict_array

st.set_page_config(page_title="Plant Leaf Disease Detection", page_icon="🌿", layout="wide")
st.title("🌿 Plant Leaf Disease Detection")
st.caption("Upload a leaf photo. The app pre-processes it and a CNN classifies it as healthy or infected.")

uploaded = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    rgb = np.array(img)

    enhanced = enhance_contrast(denoise(rgb))
    segmented, mask = segment_leaf(enhanced)
    overlay = diseased_overlay(enhanced, mask)
    ratio = diseased_area_ratio(enhanced, mask)

    c1, c2, c3, c4 = st.columns(4)
    c1.image(rgb, caption="Original", use_container_width=True)
    c2.image(enhanced, caption="Denoised + CLAHE", use_container_width=True)
    c3.image(segmented, caption="Segmented leaf", use_container_width=True)
    c4.image(overlay, caption="Suspicious regions (red)", use_container_width=True)

    st.divider()
    try:
        label, conf = predict_array(rgb)
        if label == "healthy":
            st.success(f"✅ Healthy leaf — confidence {conf:.1%}")
        else:
            st.error(f"⚠️ Infected leaf — confidence {conf:.1%}")
    except FileNotFoundError as e:
        st.warning(str(e))

    st.metric("Non-green leaf area (heuristic)", f"{ratio:.1%}")
else:
    st.info("Upload an image to begin.")
