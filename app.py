import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Undertone Finder",
    page_icon="💅🏻",
    layout="wide"
)

# ================== GLOBAL CSS ==================
st.markdown("""
<style>
.stApp {
    background-color: #fffafc;
    font-family: 'Segoe UI', sans-serif;
}
section[data-testid="stSidebar"] {
    background-color: #ffe6f0;
}
section[data-testid="stSidebar"] * {
    color: #b83280 !important;
}
.card {
    background: white;
    padding: 25px;
    border-radius: 25px;
    box-shadow: 0 8px 25px rgba(255, 77, 166, 0.15);
    margin: 20px 0;
}
.badge {
    background: #ff4da6;
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 14px;
}
.footer {
    text-align: center;
    color: #c41d7f;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("model2.h5")
    with open("label_encoder2.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, label_encoder = load_model_and_encoder()

# ================== FUNCTIONS ==================
def preprocess_image(image, size=(64, 64)):
    image = image.resize(size)
    img = np.array(image).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict(image_array):
    probs = model.predict(image_array)[0]
    idx = np.argmax(probs)
    return label_encoder.inverse_transform([idx])[0], probs[idx]

# ================== SIDEBAR ==================
menu = st.sidebar.selectbox("🌸 Menu", ["HOME", "CHECK YOUR UNDERTONE"])

# ================== HOME ==================
if menu == "HOME":
    st.markdown("## 💅🏻 Undertone Finder")
    st.markdown("""
    <div class="card">
    <span class="badge">Apa itu Undertone?</span>
    <p>Undertone adalah warna dasar alami kulit yang tidak berubah.</p>
    <ul>
        <li>Makeup lebih nyatu</li>
        <li>Baju lebih flattering</li>
        <li>Aksesori makin cocok</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.image("undertone.png", width=420)
    st.markdown("👉 Pilih menu <b>CHECK YOUR UNDERTONE</b>", unsafe_allow_html=True)

# ================== CHECK UNDERTONE ==================
else:
    st.markdown("## 🔍 Deteksi Undertone Kulit")
    tab1, tab2 = st.tabs(["📁 Upload Gambar", "📷 Kamera"])

    def show_result(pred, conf):
        pred = pred.strip().capitalize()

        st.markdown(f"""
        <div class="card">
        <span class="badge">Hasil Deteksi</span>
        <h3>Undertone kamu: <b>{pred}</b></h3>
        <p>Confidence: <b>{conf*100:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 💖 Rekomendasi Warna")

        if pred == "Cool":
            st.markdown("""
            ✔ <span style="color:#1f77b4"><b>Biru</b></span>,
            <span style="color:#6a0dad"><b>Ungu</b></span>,
            <span style="color:#7f7f7f"><b>Abu-abu</b></span>,
            <span style="color:#c0c0c0"><b>Silver</b></span>
            """, unsafe_allow_html=True)
            st.image("COOL.png", width=240)

        elif pred == "Warm":
            st.markdown("""
            ✔ <span style="color:#f1c40f"><b>Kuning</b></span>,
            <span style="color:#8b4513"><b>Coklat</b></span>,
            <span style="color:#d4af37"><b>Emas</b></span>,
            <span style="color:#808000"><b>Olive</b></span>
            """, unsafe_allow_html=True)
            st.image("WARM.png", width=240)

        else:
            st.markdown("""
            ✔ <span style="color:#d2b48c"><b>Beige</b></span>,
            <span style="color:#ffb7c5"><b>Peach</b></span>,
            <span style="color:#ff69b4"><b>Pink</b></span>,
            <span style="color:#98ff98"><b>Mint</b></span>
            """, unsafe_allow_html=True)
            st.image("NEUTRAL.png", width=240)

    # ===== UPLOAD =====
    with tab1:
        file = st.file_uploader("Upload gambar (jpg/png)", type=["jpg", "png", "jpeg"])
        if file:
            col1, col2 = st.columns([1, 1.4])
            with col1:
                img = Image.open(file).convert("RGB")
                st.image(img, width=260, caption="Gambar yang diupload")
            with col2:
                pred, conf = predict(preprocess_image(img))
                show_result(pred, conf)

    # ===== CAMERA =====
    with tab2:
        cam = st.camera_input("Ambil gambar")
        if cam:
            img = Image.open(cam).convert("RGB")
            st.image(img, width=260)
            pred, conf = predict(preprocess_image(img))
            show_result(pred, conf)

# ================== FOOTER ==================
st.markdown("""
<div class="footer">
✨ Undertone Finder · Streamlit App ✨
</div>
""", unsafe_allow_html=True)

