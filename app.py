import os
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load the trained model
MODEL_PATH = "dog_skin397.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (Update these based on your dataset)
class_labels = ["Flea allergy", "Hotspot", "Mange"]

# Remedies for each disease
remedies = {
    "Flea allergy": "Flea Allergy Dermatitis (FAD) occurs due to flea saliva, causing itching, hair loss, and scabs. Home remedies include apple cider vinegar spray, aloe vera, coconut oil, and oatmeal baths. Treatment involves flea control medications, antihistamines, medicated shampoos, and antibiotics if needed. Prevent fleas with regular grooming, cleaning, and flea preventatives.",
    "Hotspot": "Hotspots develop from allergies or moisture buildup, presenting as inflamed, oozing sores. Chamomile tea compress, betadine, and aloe vera provide relief, while treatment includes trimming fur, antiseptics, antibiotics, and E-collars to prevent licking. Keep the skin dry and address allergies to prevent hotspots.",
    "Mange": "Mange, caused by mites, leads to intense itching and hair loss. Remedies like neem oil, coconut oil, and yogurt help, but vet-prescribed ivermectin, medicated shampoos, and antibiotics may be necessary. Prevent mange with hygiene and immune-boosting diets.",
}

# Streamlit UI
st.title("🐶 Dog Skin Disease Classifier")
st.write("Upload an image of the dog's skin to predict the disease and get remedies.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))  # Resize to model input size
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Display result
    st.subheader(f"✅ Predicted Disease: {predicted_class}")
    st.write(remedies.get(predicted_class, "Consult a veterinarian for diagnosis and treatment."))
