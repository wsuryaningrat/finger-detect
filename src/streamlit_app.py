import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

# ==================================================
# PATH
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT / "src"))

from model import SiameseNetwork
from utils import get_valid_transform
from minutiae import extract_minutiae

# ==================================================
# CONFIG
# ==================================================

MODEL_PATH = ROOT / "models" / "best_model.pth"

THRESHOLD = 0.2486

IMG_SIZE = (224, 224)

# ==================================================
# DEVICE
# ==================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# ==================================================
# PAGE
# ==================================================

st.set_page_config(
    page_title="Fingerprint Verification",
    page_icon="🔍",
    layout="wide"
)

# ==================================================
# MODEL
# ==================================================

@st.cache_resource
def load_model():

    model = SiameseNetwork()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    return model


model = load_model()

transform = get_valid_transform()

# ==================================================
# PREPROCESS
# ==================================================

def normalize_image(img):

    img = img.astype(np.float32)

    mean = np.mean(img)
    std = np.std(img)

    img = (
        img - mean
    ) / (std + 1e-8)

    img = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return img.astype(np.uint8)


def apply_clahe(img):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(img)


@st.cache_data
def preprocess_bytes(file_bytes):

    img = cv2.imdecode(
        np.frombuffer(
            file_bytes,
            np.uint8
        ),
        cv2.IMREAD_GRAYSCALE
    )

    img = normalize_image(img)

    img = apply_clahe(img)

    img = cv2.resize(
        img,
        IMG_SIZE
    )

    tensor = transform(img)

    return img, tensor


# ==================================================
# HEADER
# ==================================================

st.title("🔍 Fingerprint Verification")

# ==================================================
# MAIN LAYOUT: INPUT | RESULT
# ==================================================

col_input, col_result = st.columns([1, 1.2], gap="medium")

# ===== LEFT COLUMN: INPUT =====
with col_input:
    st.subheader("📁 INPUT")
    
    file1 = st.file_uploader("Fingerprint A", type=["bmp", "jpg", "jpeg", "png"], key="file_a", label_visibility="collapsed")
    file2 = st.file_uploader("Fingerprint B", type=["bmp", "jpg", "jpeg", "png"], key="file_b", label_visibility="collapsed")
    
    if file1 and file2:
        img1, tensor1 = preprocess_bytes(file1.getvalue())
        img2, tensor2 = preprocess_bytes(file2.getvalue())
        
        # Compact preview (side by side)
        p1, p2 = st.columns(2, gap="small")
        with p1:
            st.image(img1)
        with p2:
            st.image(img2)
        
        verify_btn = st.button("🚀 VERIFY", type="primary", use_container_width=True)
    else:
        verify_btn = False
        img1 = img2 = tensor1 = tensor2 = None

# ===== RIGHT COLUMN: RESULT =====
with col_result:
    st.subheader("📊 RESULT")
    
    if file1 and file2 and verify_btn:
        start_total = time.time()
        tensor1 = tensor1.unsqueeze(0).to(DEVICE)
        tensor2 = tensor2.unsqueeze(0).to(DEVICE)
        
        start_inf = time.time()
        with torch.no_grad():
            emb1, emb2 = model(tensor1, tensor2)
            distance = F.pairwise_distance(emb1, emb2).item()
        
        inference_time = time.time() - start_inf
        total_time = time.time() - start_total
        similarity = 1 / (1 + distance)
        is_match = distance < THRESHOLD
        
        # Display metrics in compact format
        st.markdown(f"**Similarity:** {similarity*100:.1f}%")
        st.markdown(f"**Distance:** {distance:.4f}")
        status_text = "✅ MATCH" if is_match else "❌ MISMATCH"
        st.markdown(f"**Decision:** {status_text}")
        
        st.divider()
        
        if is_match:
            st.success("Fingerprints Match", icon="✅")
        else:
            st.error("Fingerprints Do Not Match", icon="❌")
        
        with st.expander("ℹ️ Verification Details", expanded=True):
            st.latex(r"similarity = \frac{1}{1 + distance}")
            st.write(f"Inference: {inference_time:.3f}s | Total: {total_time:.3f}s | Device: {DEVICE}")
    else:
        st.info("👆 Upload images and click VERIFY", icon="ℹ️")

# ==================================================
# EXPLAINABILITY SECTION
# ==================================================

if file1 and file2 and verify_btn:
    st.divider()
    st.subheader("🔍 EXPLAINABILITY")
    
    with st.spinner("Extracting minutiae..."):
        result1 = extract_minutiae(img1)
        result2 = extract_minutiae(img2)
    
    exp_col1, exp_col2 = st.columns(2, gap="medium")
    
    with exp_col1:
        st.markdown("**Fingerprint A**")
        st.image(result1["overlay"], width=250)
        e1, b1 = len(result1["endings"]), len(result1["bifurcations"])
        st.markdown(f"Endings : {e1} | Bifurcations : {b1}")
    
    with exp_col2:
        st.markdown("**Fingerprint B**")
        st.image(result2["overlay"], width=250)
        e2, b2 = len(result2["endings"]), len(result2["bifurcations"])
        st.markdown(f"Endings : {e2} | Bifurcations : {b2}")

# ==================================================
# FOOTER
# ==================================================

st.divider()
st.caption("Designed & Developed by WHS • Telkom University")