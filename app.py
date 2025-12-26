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

# ================= LOAD MODEL =================
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("model2.h5")
    with open("label_encoder2.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model_and_encoder()

# ================= PREPROCESS =================
def preprocess_image(image, target_size=(64, 64)):
    image = image.resize(target_size)
    img = np.array(image).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ================= PREDICT =================
def predict(image_array):
    preds = model.predict(image_array)[0]
    idx = np.argmax(preds)
    label = label_encoder.inverse_transform([idx])[0]
    conf = preds[idx]
    return label, conf

# ================= SIDEBAR =================
menu = ["HOME", "CHECK YOUR UNDERTONE HERE"]
choice = st.sidebar.selectbox("Navigasi", menu)

# ================= HOME =================
if choice == "HOME":
    st.title("💅🏻 Undertone Finder")
    st.markdown("""
    **Kenali undertone kulitmu dengan AI ✨**

    Undertone adalah warna dasar alami kulit yang **tidak berubah**
    meskipun kulitmu menjadi lebih gelap atau terang.

    ### 💡 Jenis Undertone
    - ❄️ **Cool** → kebiruan / pink
    - 🔥 **Warm** → kekuningan / keemasan
    - 🌿 **Neutral** → campuran cool & warm
    """)

    st.image("undertone.png", use_container_width=True)
    st.info("👉 Gunakan menu samping untuk mulai deteksi")

# ================= DETECTION =================
elif choice == "CHECK YOUR UNDERTONE HERE":
    st.title("🔍 Deteksi Undertone Kulit")
    st.caption("Upload foto nadi atau gunakan kamera untuk hasil terbaik")

    tab1, tab2 = st.tabs(["📁 Upload Gambar", "📷 Kamera Realtime"])

    # ---------- UPLOAD ----------
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload gambar nadi (jpg, png)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            col1, col2 = st.columns([1, 1])

            with col1:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Gambar yang diupload", width=280)

            with col2:
                processed_img = preprocess_image(image)
                predicted_class, confidence = predict(processed_img)

                st.markdown("### ✨ Hasil Prediksi")
                st.success(f"**{predicted_class} Undertone**")
                st.metric("Confidence", f"{confidence*100:.2f}%")

            st.divider()

            if predicted_class == "Cool":
                st.write("💙 **Rekomendasi warna:** Biru, Ungu, Abu-abu, Silver")
                st.image("COOL.png", width=250)
            elif predicted_class == "Warm":
                st.write("💛 **Rekomendasi warna:** Kuning, Coklat, Emas, Olive")
                st.image("WARM.png", width=250)
            else:
                st.write("💚 **Rekomendasi warna:** Beige, Peach, Pink, Mint")
                st.image("NEUTRAL.png", width=250)

    # ---------- CAMERA ----------
    with tab2:
        camera_image = st.camera_input("Ambil gambar nadi")

        if camera_image:
            col1, col2 = st.columns([1, 1])

            with col1:
                image = Image.open(camera_image).convert("RGB")
                st.image(image, caption="Hasil kamera", width=280)

            with col2:
                processed_img = preprocess_image(image)
                predicted_class, confidence = predict(processed_img)

                st.markdown("### ✨ Hasil Prediksi")
                st.success(f"**{predicted_class} Undertone**")
                st.metric("Confidence", f"{confidence*100:.2f}%")

            st.divider()

            if predicted_class == "Cool":
                st.write("💙 **Rekomendasi warna:** Biru, Ungu, Abu-abu, Silver")
                st.image("COOL.png", width=250)
            elif predicted_class == "Warm":
                st.write("💛 **Rekomendasi warna:** Kuning, Coklat, Emas, Olive")
                st.image("WARM.png", width=250)
            else:
                st.write("💚 **Rekomendasi warna:** Beige, Peach, Pink, Mint")
                st.image("NEUTRAL.png", width=250)

# ================= FOOTER =================
st.divider()
st.caption("✨ Undertone Finder • AI Powered • Streamlit App")
