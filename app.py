import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import pickle


# ====== Load Model dan Encoder ======
# Load Model dan LabelEncoder
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("model2.h5")
    with open("label_encoder2.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model_and_encoder()

# ====== Fungsi Preprocessing ======
# ====== Fungsi Preprocessing ======
def preprocess_image(image, target_size=(64, 64)):
    image = image.resize(target_size)
    img = np.array(image).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ====== Fungsi Prediksi ======
def predict(image_array):
    predictions = model.predict(image_array)[0]  # karena image-nya cuma satu
    predicted_index = np.argmax(predictions)
    predicted_class = label_encoder.inverse_transform([predicted_index])[0]
    confidence = predictions[predicted_index]
    return predicted_class, confidence


# ====== Sidebar Menu ======
menu = ["HOME", "CHECK YOUR UNDERTONE HERE"]
choice = st.sidebar.selectbox("Navigasi", menu)

if choice == "HOME":
    st.title("Welcome to Undertone Finder💅🏻")
    st.markdown("""
    ## Apa itu Undertone?  
    Undertone adalah warna dasar alami kulit yang tidak berubah meskipun warna kulitmu berubah karena paparan matahari.  
    Mengetahui undertone kulitmu bisa membantu kamu memilih warna pakaian, makeup, dan aksesori yang paling cocok.

    ### Jenis Undertone:
    - **Cool** - Nada kebiruan atau merah muda
    - **Warm** - Nada kekuningan atau keemasan
    - **Neutral** - Campuran antara cool dan warm
    """)
    st.image("undertone.png", use_container_width=True)
    st.markdown("Yuk mulai deteksi di menu sebelah! 👈🏻")

elif choice == "CHECK YOUR UNDERTONE HERE":
    st.title("🔍 Deteksi Undertone Kulit")
    st.write("Pilih metode input gambar:")

    tab1, tab2 = st.tabs(["📁 Upload File", "📷 Kamera Realtime"])

    with tab1:
        uploaded_file = st.file_uploader("Upload gambar nadi (jpg, png)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar yang diupload", use_container_width=True)

            processed_img = preprocess_image(image)
            predicted_class, confidence = predict(processed_img)

            st.markdown("### Hasil Prediksi")
            st.success(f"Undertone kamu: **{predicted_class}**")
            st.info(f"Tingkat keyakinan model: **{confidence*100:.2f}%**")

            if predicted_class == "Cool":
                st.write("- Warna cocok: Biru, Ungu, Abu-abu, Silver")
                st.image("COOL.png", caption="Palet Warna untuk Undertone Cool", width=300)
            elif predicted_class == "Warm":
                st.write("- Warna cocok: Kuning, Coklat, Emas, Hijau Zaitun")
                st.image("WARM.png", caption="Palet Warna untuk Undertone Warm", width=300)
            else:
                st.write("- Warna cocok: Beige, Peach, Merah Muda, Mint")
                st.image("NEUTRAL.png", caption="Palet Warna untuk Undertone Neutral", width=300)

    with tab2:
        st.write("Ambil gambar dari kamera (gunakan tombol di bawah)")
        camera_image = st.camera_input("Ambil Gambar dari Kamera")

        if camera_image:
            image = Image.open(camera_image).convert("RGB")
            st.image(image, caption="Gambar dari Kamera", use_container_width=True)

            processed_img = preprocess_image(image)
            predicted_class, confidence = predict(processed_img)

            st.markdown("### Hasil Prediksi")
            st.success(f"Undertone kamu: **{predicted_class}**")
            st.info(f"Tingkat keyakinan model: **{confidence*100:.2f}%**")

            if predicted_class == "Cool":
                st.write("- Warna cocok: Biru, Ungu, Abu-abu, Silver")
                st.image("COOL.png", caption="Palet Warna untuk Undertone Cool", width=300)
            elif predicted_class == "Warm":
                st.write("- Warna cocok: Kuning, Coklat, Emas, Hijau Zaitun")
                st.image("WARM.png", caption="Palet Warna untuk Undertone Warm", width=300)
            else:
                st.write("- Warna cocok: Beige, Peach, Merah Muda, Mint")
                st.image("NEUTRAL.png", caption="Palet Warna untuk Undertone Neutral", width=300)
