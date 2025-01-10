import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('KIRAnet.h5')

label_map = {
    0: "shayad burger hai",
    1: "shayad naan hai",
    2: "shayad chai hai",
    3: "shayad roti hai",
    4: "shayad chole bhature hai",
    5: "shayad dal hai",
    6: "shayad dhokla hai",
    7: "shayad namkeen hai",
    8: "shayad idly hai",
    9: "shayad jalebi hai",
    10: "shayad spring Roll hai",
    11: "shayad koi sabji hai",
    12: "shayad kulfi hai",
    13: "shayad dosa hai",
    14: "shayad momo's hai"
    16: "shayad pakode hai",
    15: "shayad puchke hai",
    17: "shayad pav Bhaji hai",
    18: "shayad pizza hai",
    19: "shayad samosa hai"
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

    st.write(f"KIRAnet ke hisaab se ye {predicted_label})

    st.write("### Probabilities:")
    for idx, probability in enumerate(probabilities):
        st.write(f"{label_map[idx]}: {probability * 100:.2f}%")
