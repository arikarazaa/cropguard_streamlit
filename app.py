"""
app.py — CropGuard Streamlit web application.
Works in two modes:
  DEMO MODE  — no model needed, shows realistic simulated predictions.
  MODEL MODE — set MODEL_PATH env var or place best_model.pth in models/.

Run locally:  streamlit run app.py  ->  http://localhost:8501
Deploy:       push to Streamlit Community Cloud
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropGuard — AI Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Risk cards — dark-mode safe with explicit text colours */
    .risk-critical { background:#3a0000; border-left:4px solid #ff5252; padding:14px 16px; border-radius:8px; color:#ffcccc !important; }
    .risk-high     { background:#3a1500; border-left:4px solid #ff6e40; padding:14px 16px; border-radius:8px; color:#ffe0cc !important; }
    .risk-moderate { background:#2e2500; border-left:4px solid #ffd600; padding:14px 16px; border-radius:8px; color:#fff9c4 !important; }
    .risk-low      { background:#0d2600; border-left:4px solid #76ff03; padding:14px 16px; border-radius:8px; color:#ccff90 !important; }
    .risk-none     { background:#002200; border-left:4px solid #00e676; padding:14px 16px; border-radius:8px; color:#b9f6ca !important; }
    .risk-critical *, .risk-high *, .risk-moderate *, .risk-low *, .risk-none * { color: inherit !important; }

    /* Prevent long disease names truncating in metric cards */
    [data-testid="stMetricValue"] { font-size:1.05rem !important; word-break:break-word; white-space:normal !important; line-height:1.4; }
</style>
""", unsafe_allow_html=True)

# ── Model path ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pth")

# ── Load predictor (cached — runs only once per session) ──────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_predictor():
    from src.inference import CropGuardPredictor
    checkpoint = MODEL_PATH if os.path.exists(MODEL_PATH) else None
    return CropGuardPredictor(checkpoint_path=checkpoint)

predictor = load_predictor()
DEMO_MODE = predictor.demo_mode

# ── Demo-mode simulation (realistic fixed outputs per class) ──────────────────
DEMO_RESULTS = {
    "Tomato Late Blight": {
        "disease": "Tomato___Late_blight", "confidence": 0.961,
        "risk": {"level": "Critical", "yield_loss_pct": 62,
                 "action": "Apply copper-based fungicide immediately. Remove infected leaves.",
                 "act_within_hours": 24},
        "top_k": [("Tomato___Late_blight", 0.961), ("Tomato___Early_blight", 0.023),
                  ("Tomato___Septoria_leaf_spot", 0.010), ("Tomato___healthy", 0.004),
                  ("Tomato___Bacterial_spot", 0.002)],
    },
    "Tomato Healthy": {
        "disease": "Tomato___healthy", "confidence": 0.994,
        "risk": {"level": "None", "yield_loss_pct": 0,
                 "action": "Healthy crop — continue monitoring.", "act_within_hours": None},
        "top_k": [("Tomato___healthy", 0.994), ("Tomato___Early_blight", 0.003),
                  ("Tomato___Late_blight", 0.002), ("Tomato___Leaf_Mold", 0.001),
                  ("Tomato___Bacterial_spot", 0.000)],
    },
    "Corn Grey Leaf Spot": {
        "disease": "Corn___Cercospora_leaf_spot_Gray_leaf_spot", "confidence": 0.942,
        "risk": {"level": "Moderate", "yield_loss_pct": 24,
                 "action": "Apply triazole fungicide. Improve air circulation.",
                 "act_within_hours": 72},
        "top_k": [("Corn___Cercospora_leaf_spot_Gray_leaf_spot", 0.942),
                  ("Corn___Northern_Leaf_Blight", 0.034), ("Corn___Common_rust", 0.014),
                  ("Corn___healthy", 0.008), ("Tomato___Late_blight", 0.002)],
    },
}

def format_name(raw: str) -> str:
    """'Tomato___Late_blight' -> 'Tomato — Late blight'"""
    parts = raw.split("___", 1)
    if len(parts) == 2:
        return f"{parts[0]} — {parts[1].replace('_', ' ')}"
    return raw.replace("_", " ")

def make_demo_gradcam(pil_image: Image.Image) -> Image.Image:
    """Generate a plausible-looking Grad-CAM overlay without a real model."""
    import cv2
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, (224, 224))
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = int(w * 0.45), int(h * 0.4)
    heatmap = np.exp(-((x - cx) ** 2 / (w * 0.18) ** 2 + (y - cy) ** 2 / (h * 0.2) ** 2))
    cx2, cy2 = int(w * 0.65), int(h * 0.62)
    heatmap += 0.6 * np.exp(-((x - cx2) ** 2 / (w * 0.12) ** 2 + (y - cy2) ** 2 / (h * 0.14) ** 2))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    hm_col = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_col = cv2.cvtColor(hm_col, cv2.COLOR_BGR2RGB)
    overlay = (0.45 * hm_col + 0.55 * img).astype(np.uint8)
    return Image.fromarray(overlay)

def run_predict(image: Image.Image, demo_scenario: str) -> dict:
    """Run prediction — real model or demo simulation."""
    if DEMO_MODE:
        result = DEMO_RESULTS.get(demo_scenario, list(DEMO_RESULTS.values())[0]).copy()
        result["gradcam_pil"] = make_demo_gradcam(image)
        result["demo_mode"] = True
    else:
        result = predictor.predict(image)
        result["demo_mode"] = False
    return result

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 CropGuard")
    st.markdown("**EfficientNet-B3 + Grad-CAM**")
    st.divider()

    uploaded_file = st.file_uploader(
        "📷 Upload a leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear photo of a single leaf",
    )

    demo_scenario = None
    if DEMO_MODE:
        st.divider()
        demo_scenario = st.radio(
            "🎬 Demo scenario",
            options=list(DEMO_RESULTS.keys()),
            index=0,
            help="Choose which disease to simulate (demo mode only)",
        )
        st.info("⚠️ **Demo mode** — no trained model found.\nPredictions are realistic simulations.")
    else:
        st.success("✅ Model loaded from checkpoint")

    analyse_btn = st.button(
        "🔍 Analyse Leaf",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None),
    )

    st.divider()
    st.markdown("""
**To use a real model:**
1. Train via `notebooks/CropGuard_Training.ipynb`
2. Place `best_model.pth` in `models/`
3. Restart the app
    """)

# ── Main header ────────────────────────────────────────────────────────────────
st.title("🌿 CropGuard — AI Crop Disease Detection")
st.caption("EfficientNet-B3 · PlantVillage · 38 disease classes · 97.2% accuracy")

RISK_EMOJI   = {"None": "✅", "Low": "🟡", "Moderate": "🟠", "High": "🔴", "Critical": "🚨"}
RISK_CSS_MAP = {
    "None": "risk-none", "Low": "risk-low", "Moderate": "risk-moderate",
    "High": "risk-high", "Critical": "risk-critical",
}

# ── Welcome screen (no image yet) ─────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
### How it works
1. **Upload** a leaf photo using the sidebar panel
2. **Select** a demo scenario (demo mode) or use your trained model
3. Click **Analyse Leaf** — or it runs automatically on upload
4. View the **disease prediction**, **Grad-CAM saliency map**, and **risk assessment**
        """)
    with col_r:
        st.markdown("""
### Supported crops
🍅 Tomato · 🥔 Potato · 🌽 Corn · 🍎 Apple
🍇 Grape · 🍑 Peach · 🫑 Pepper · 🍓 Strawberry
🫐 Blueberry · 🍊 Orange · 🌱 Soybean · and more
        """)
    st.stop()

# ── Prepare image ──────────────────────────────────────────────────────────────
pil_image = Image.open(uploaded_file).convert("RGB")

# Track state to auto-run when upload or scenario changes
if "last_file"     not in st.session_state: st.session_state.last_file     = None
if "last_scenario" not in st.session_state: st.session_state.last_scenario = None
if "result"        not in st.session_state: st.session_state.result        = None

file_changed     = st.session_state.last_file     != uploaded_file.name
scenario_changed = st.session_state.last_scenario != demo_scenario

if analyse_btn or file_changed or scenario_changed:
    with st.spinner("Analysing leaf…"):
        st.session_state.result        = run_predict(pil_image, demo_scenario)
        st.session_state.last_file     = uploaded_file.name
        st.session_state.last_scenario = demo_scenario

result = st.session_state.result

if result is None:
    st.info("Upload a leaf image and click **Analyse Leaf**.")
    st.stop()

# ── Unpack result ──────────────────────────────────────────────────────────────
disease     = format_name(result["disease"])
conf        = result["confidence"]
risk        = result["risk"]
gradcam_pil = result["gradcam_pil"]
emoji       = RISK_EMOJI.get(risk["level"], "⚠️")
css_class   = RISK_CSS_MAP.get(risk["level"], "risk-moderate")

# ── Image + Grad-CAM ───────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("📷 Input Image")
    st.image(pil_image, use_container_width=True)

with col2:
    st.subheader("🔥 Grad-CAM Saliency Map")
    st.image(gradcam_pil, use_container_width=True)
    if result.get("demo_mode"):
        st.caption("⚠️ Demo mode — Grad-CAM is simulated. Use a trained checkpoint for real explanations.")
    else:
        st.caption("🔴 Red = high activation  ·  🔵 Blue = low  ·  Shows which leaf regions drove the prediction (Grad-CAM)")

# ── Metric cards ───────────────────────────────────────────────────────────────
st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("🦠 Detected",        disease)
m2.metric("🎯 Confidence",      f"{conf * 100:.1f}%")
m3.metric("📊 Risk Level",      f"{emoji} {risk['level']}")
m4.metric("🌾 Est. Yield Loss", f"{risk['yield_loss_pct']}%")

# ── Risk action banner ─────────────────────────────────────────────────────────
act_within = (
    f"Act within <strong>{risk['act_within_hours']} hours</strong>"
    if risk["act_within_hours"] else "No urgent action required"
)
st.markdown(f"""
<div class="{css_class}" style="margin-top:12px">
  <strong>{emoji} Recommended Action</strong><br><br>
  {risk['action']}<br><br>
  ⏱️ {act_within}
</div>
""", unsafe_allow_html=True)

# ── Top-5 predictions ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Top-5 Predictions")

rank_icons = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
for i, (name, prob) in enumerate(result["top_k"], 1):
    col_name, col_bar, col_pct = st.columns([3, 4, 1])
    with col_name:
        st.markdown(f"{rank_icons[i-1]} **{format_name(name)}**")
    with col_bar:
        st.progress(float(prob))
    with col_pct:
        st.markdown(f"`{prob * 100:.1f}%`")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Dataset:** PlantVillage (Hughes & Salathé, 2015) · "
    "**Architecture:** EfficientNet-B3 (timm) · "
    "**Explainability:** Grad-CAM (Selvaraju et al., ICCV 2017) · "
    "**Research alignment:** AgroRisk · OPTIcut · De Montfort University · "
    "**Author:** Muhammad Amir Raza"
)
