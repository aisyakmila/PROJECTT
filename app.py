import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Undertone Finder",
    page_icon="💅🏻",
    layout="centered"
)

# ================= CSS =================
st.markdown("""
<style>

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #ffe6f0 !important;
}

/* BUTTON */
.stButton > button {
    background-color: #ff4da6;
    color: white;
    border-radius: 30px;
    padding: 10px 20px;
    font-weight: bold;
}

/* CARD */
.card {
    background-color: white;
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0 8px 20px rgba(255,77,166,0.2);
    margin-bottom: 20px;
}

/* IMAGE */
img {
    border-radius: 20px;
}

/* TITLE */
h1, h2, h3 {
    color: #b83280;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("model2.h5")
    with open("label_encoder2.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, label_encoder = load_model_and_encoder()

# ================= PREPROCESS =================
def preprocess_image(image, size=(64, 64)):
    image = image.resize(size)
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(image):
    preds = model.predict(image)[0]
    idx = np.argmax(preds)
    return label_encoder.inverse_transform([idx])[0], preds[idx]

# ================= SIDEBAR =================
st.sidebar.title("💗 Undertone Finder")
menu = st.sidebar.radio(
    "Menu",
    ["Home", "Check Undertone"]
)

# ================= HOME =================
if menu == "Home":
    st.title("✨ Welcome to Undertone Finder")

    st.markdown("""
    <div class="card">
    Undertone adalah warna dasar alami kulit yang **tidak berubah** walaupun kulitmu menjadi lebih terang atau gelap.
    
    Mengetahui undertone membantu kamu memilih:
    - 💄 Makeup
    - 👗 Warna pakaian
    - 💍 Aksesori
    </div>
    """, unsafe_allow_html=True)

    st.image("assets/undertone.png", use_container_width=True)
    st.info("👉 Klik menu **Check Undertone** di sidebar")

# ================= CHECK =================
else:
    st.title("🔍 Check Your Undertone")

    tab1, tab2 = st.tabs(["📁 Upload", "📷 Camera"])

    with tab1:
        file = st.file_uploader("Upload foto nadi (jpg/png)", type=["jpg","png","jpeg"])
        if file:
            image = Image.open(file).convert("RGB")
            st.image(image, width=300)

            img = preprocess_image(image)
            tone, conf = predict(img)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.success(f"Undertone kamu: **{tone}**")
            st.info(f"Confidence: **{conf*100:.2f}%**")

            if tone == "Cool":
                st.image("assets/COOL.png", width=300)
                st.write("✔ Cocok warna biru, ungu, silver")
            elif tone == "Warm":
                st.image("assets/WARM.png", width=300)
                st.write("✔ Cocok warna emas, coklat, olive")
            else:
                st.image("assets/NEUTRAL.png", width=300)
                st.write("✔ Cocok hampir semua warna")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        cam = st.camera_input("Ambil gambar")
        if cam:
            image = Image.open(cam).convert("RGB")
            st.image(image, width=300)

            img = preprocess_image(image)
            tone, conf = predict(img)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.success(f"Undertone kamu: **{tone}**")
            st.info(f"Confidence: **{conf*100:.2f}%**")
            st.markdown("</div>", unsafe_allow_html=True)
