import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('KIRAnet.h5')

label_map = {
    0: "Shayad Burger hai",
    1: "Shayad Naan hai",
    2: "Shayad Chai hai",
    3: "Shayad Roti hai",
    4: "Shayad Chole Bhature hai",
    5: "Shayad Dal hai",
    6: "Shayad Dhokla hai",
    7: "Shayad Namkeen hai",
    8: "Shayad Idly hai",
    9: "Shayad Jalebi hai",
    10: "Shayad Spring Roll hai",
    11: "Shayad koi Sabji hai",
    12: "Shayad Kulfi hai",
    13: "Shayad Dosa hai",
    14: "Shayad Pakode hai",
    15: "Shayad Puchke hai",
    16: "Shayad Pav Bhaji hai",
    17: "Shayad Pizza hai",
    18: "Shayad Samosa hai"
}


def preprocess_image(image, target_size=(224, 224)):

    if len(image.shape) == 2 or image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image_resized = cv2.resize(image, target_size)
    return np.expand_dims(image_resized, axis=0)


def predict_label(image, model, label_map):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)[0]
    predicted_index = np.argmax(predictions)
    predicted_label = label_map[predicted_index]
    confidence = predictions[predicted_index] * 100

    return predicted_label, confidence, predictions


st.title("Food Dish Classifier 🍽️")
st.write("Upload an image of a food dish to classify it.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

    predicted_label, confidence, probabilities = predict_label(image_rgb, model, label_map)

    st.write(f"**KIRAnet ke hisaab se ye {predicted_label} aur hone ke chances {confidence:.2f}% hai.**")

    st.write("### Probabilities:")
    for idx, probability in enumerate(probabilities):
        st.write(f"{label_map[idx]}: {probability * 100:.2f}%")