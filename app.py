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

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #ffe6f0;
    padding-top: 20px;
}

/* SIDEBAR TEXT */
section[data-testid="stSidebar"] * {
    color: #b83280 !important;
    font-weight: 500;
}

/* SELECTBOX */
div[data-baseweb="select"] > div {
    background-color: #fff0f6;
    border-radius: 14px;
    border: 1px solid #ffadd2;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(90deg, #ff85c0, #ff4da6);
    color: white;
    border-radius: 30px;
    border: none;
    padding: 0.6em 1.4em;
    font-size: 16px;
}

/* BUTTON HOVER */
.stButton > button:hover {
    background: linear-gradient(90deg, #ff4da6, #c41d7f);
}

/* CARD */
.card {
    background: white;
    padding: 25px;
    border-radius: 25px;
    box-shadow: 0 8px 25px rgba(255, 77, 166, 0.15);
    margin-top: 20px;
}

/* BADGE */
.badge {
    display: inline-block;
    background: #ff4da6;
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 14px;
    margin-bottom: 10px;
}

/* IMAGE */
img {
    border-radius: 18px;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #c41d7f;
    margin-top: 50px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("model2.h5")
    with open("label_encoder2.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model_and_encoder()

# ================== FUNCTIONS ==================
def preprocess_image(image, target_size=(64, 64)):
    image = image.resize(target_size)
    img = np.array(image).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict(image_array):
    probs = model.predict(image_array)[0]
    idx = np.argmax(probs)
    return label_encoder.inverse_transform([idx])[0], probs[idx]

# ================== SIDEBAR ==================
menu = st.sidebar.selectbox(
    "🌸 Navigasi",
    ["HOME", "CHECK YOUR UNDERTONE"]
)

# ================== HOME ==================
if menu == "HOME":
    st.markdown("## 💅🏻 Undertone Finder")
    st.markdown("""
    <div class="card">
    <span class="badge">Apa itu Undertone?</span>
    <p>
    Undertone adalah warna dasar alami kulitmu yang tidak berubah meskipun kulitmu
    menjadi lebih gelap atau terang.
    </p>

    <b>Kenapa penting?</b>
    <ul>
        <li>Makeup jadi lebih nyatu</li>
        <li>Warna baju lebih flattering</li>
        <li>Perhiasan terlihat lebih cocok</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.image("undertone.png", width=450)
    st.markdown("👉 Pilih menu <b>CHECK YOUR UNDERTONE</b> di sidebar", unsafe_allow_html=True)

# ================== CHECK UNDERTONE ==================
else:
    st.markdown("## 🔍 Deteksi Undertone Kulit")

    tab1, tab2 = st.tabs(["📁 Upload Gambar", "📷 Kamera"])

    # ===== UPLOAD =====
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload gambar nadi (jpg / png)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            col1, col2 = st.columns([1, 1.4])

            with col1:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, width=260, caption="Gambar yang diupload")

            with col2:
                processed = preprocess_image(image)
                pred, conf = predict(processed)

                st.markdown(f"""
                <div class="card">
                <span class="badge">Hasil Deteksi</span>
                <h3>Undertone kamu: <b>{pred}</b></h3>
                <p>Tingkat keyakinan model: <b>{conf*100:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 💖 Rekomendasi Warna")
                pred = pred.strip().capitalize()
                if pred == "Cool":
                    st.write("✔ Biru, Ungu, Abu-abu, Silver")
                    st.image("COOL.png", width=260)
                elif pred == "Warm":
                    st.write("✔ Kuning, Coklat, Emas, Olive")
                    st.image("WARM.png", width=260)
                else:
                    st.write("✔ Beige, Peach, Pink, Mint")
                    st.image("NEUTRAL.png", width=260)

    # ===== CAMERA =====
    with tab2:
        camera_image = st.camera_input("Ambil gambar dari kamera")

        if camera_image:
            image = Image.open(camera_image).convert("RGB")
            st.image(image, width=260)

            processed = preprocess_image(image)
            pred, conf = predict(processed)

            st.markdown(f"""
            <div class="card">
            <h3>✨ Undertone kamu: <b>{pred}</b></h3>
            <p>Confidence: <b>{conf*100:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown("""
<div class="footer">
✨ Undertone Finder · Streamlit App ✨
</div>
""", unsafe_allow_html=True)

