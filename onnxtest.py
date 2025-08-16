import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image

# ---------------------------
# Load ONNX Model
# ---------------------------
MODEL_PATH = "dog_skin397.onnx"
session = ort.InferenceSession(MODEL_PATH)

# Get model input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ---------------------------
# Class Labels & Remedies
# ---------------------------
class_labels = ["Flea allergy", "Hotspot", "Mange"]

remedies = {
    "Flea allergy": "Flea Allergy Dermatitis (FAD) occurs due to flea saliva, causing itching, hair loss, and scabs. "
                    "Home remedies include apple cider vinegar spray, aloe vera, coconut oil, and oatmeal baths. "
                    "Treatment involves flea control medications, antihistamines, medicated shampoos, and antibiotics if needed. "
                    "Prevent fleas with regular grooming, cleaning, and flea preventatives.",
    "Hotspot": "Hotspots develop from allergies or moisture buildup, presenting as inflamed, oozing sores. "
               "Chamomile tea compress, betadine, and aloe vera provide relief, while treatment includes trimming fur, antiseptics, "
               "antibiotics, and E-collars to prevent licking. Keep the skin dry and address allergies to prevent hotspots.",
    "Mange": "Mange, caused by mites, leads to intense itching and hair loss. "
             "Remedies like neem oil, coconut oil, and yogurt help, but vet-prescribed ivermectin, medicated shampoos, "
             "and antibiotics may be necessary. Prevent mange with hygiene and immune-boosting diets.",
}

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🐶 Dog Skin Disease Classifier (ONNX Model)")
st.write("Upload a dog skin image to predict the disease and get remedies.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img_resized = image.resize((224, 224))   # Resize to model input
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Run inference
    prediction = session.run([output_name], {input_name: img_array})[0]
    predicted_class = class_labels[np.argmax(prediction)]

    # Show results
    st.subheader(f"✅ Predicted Disease: {predicted_class}")
    st.write("### 🩺 Suggested Remedy:")
    st.write(remedies.get(predicted_class, "Consult a veterinarian for diagnosis and treatment."))
