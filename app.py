import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle

# ======================================================
# 🎀 PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Undertone Finder",
    page_icon="💅🏻",
    layout="wide"
)

# ======================================================
# 🎨 CUSTOM CSS (PINK AESTHETIC)
# ======================================================
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background-color: #ffe6f0;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {
    color: #d63384;
}

div[data-baseweb="select"] > div {
    background-color: #fff0f6;
    border-radius: 10px;
}

.stButton button {
    background-color: #ff85c0;
    color: white;
    border-radius: 20px;
}

.stButton button:hover {
    background-color: #ff4da6;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 🤖 LOAD MODEL & ENCODER
# ======================================================
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("model2.h5")
    with open("label_encoder2.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, label_encoder = load_model_and_encoder()

# ======================================================
# 🖼️ PREPROCESS IMAGE
# ======================================================
def preprocess_image(image, target_size=(64, 64)):
    image = image.resize(target_size)
    img = np.array(image).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ======================================================
# 🔮 PREDICT
# ======================================================
def predict(image_array):
    probs = model.predict(image_array)[0]
    idx = np.argmax(probs)
    label = label_encoder.inverse_transform([idx])[0]
    conf = probs[idx]
    return label, conf

# ======================================================
# 📚 RECOMMENDATION DATA
# ======================================================
recommendation = {
    "Cool": {
        "desc": "Undertone cool memiliki nuansa kebiruan atau pink sehingga cocok dengan warna dingin.",
        "makeup": ["Rose", "Berry", "Mauve", "Plum"],
        "outfit": ["Biru", "Ungu", "Abu-abu", "Silver"],
        "img": "COOL.png"
    },
    "Warm": {
        "desc": "Undertone warm memiliki nuansa kekuningan yang tampak cerah dengan warna hangat.",
        "makeup": ["Coral", "Peach", "Terracotta"],
        "outfit": ["Kuning", "Coklat", "Olive", "Gold"],
        "img": "WARM.png"
    },
    "Neutral": {
        "desc": "Undertone neutral seimbang antara hangat dan dingin, fleksibel untuk banyak warna.",
        "makeup": ["Peach", "Soft Pink", "Nude"],
        "outfit": ["Beige", "Mint", "Dusty Pink", "Cream"],
        "img": "NEUTRAL.png"
    }
}

# ======================================================
# 📌 SIDEBAR
# ======================================================
menu = ["HOME", "CHECK YOUR UNDERTONE HERE"]
choice = st.sidebar.selectbox("Navigasi", menu)

# ======================================================
# 🏠 HOME
# ======================================================
if choice == "HOME":
    st.markdown("## 💅🏻 Welcome to **Undertone Finder**")
    st.markdown("""
    **Undertone Finder** adalah aplikasi berbasis AI yang membantu kamu  
    mengetahui undertone kulit hanya dari **gambar nadi tangan**.

    ✨ Dengan mengetahui undertone, kamu bisa:
    - Memilih warna makeup yang tepat  
    - Menyesuaikan outfit & aksesoris  
    - Tampil lebih glowing & confident  
    """)

    st.image("undertone.png", use_container_width=True)
    st.success("👉 Yuk cek undertone kamu di menu samping!")

# ======================================================
# 🔍 CHECK UNDE
