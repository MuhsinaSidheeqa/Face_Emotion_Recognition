import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64

MODEL_PATH = "fer_cnn_project.keras"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    st.error("🚨 Model file not found! Please check the path.")
    st.stop()

# ---- Load Model with Error Ha

# ---- Define Emotion Classes ----
class_names = ['😡 Angry', '🤢 Disgust', '😨 Fear', '😊 Happy', '😐 Neutral', '😢 Sad', '😲 Surprise']

# ---- Emotion Detection Function ----
def detect_emotion(img):
    try:
        img = img.resize((48, 48)).convert('L')  # Resize & Convert to Grayscale
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = round(prediction[0][predicted_index] * 100, 2)

        return predicted_class, confidence
    except Exception as e:
        return None, f"Error: {str(e)}"

def add_background_image(image_file):
    with open(image_file, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode()

    # Inject CSS for background image
    css_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css_style, unsafe_allow_html=True)


def main():
    page = st.sidebar.radio("📌 Navigation", ["🏠Home", "📷Upload Image", "ℹ️About"])
    add_background_image('faces.jpg')
    if page == '🏠Home':
        st.title(":blue[🎭 Facial Emotion Recognition]")
        st.markdown("## Welcome!")
        st.write("  ")
        st.write("  ")
        st.write("This web application uses deep learning to recognize **emotions** from facial expressions in images.")
        st.write("  ")
        st.write("It  can  classify  emotions  like  **Happy** , **Sad** , **Angry** , **Fear** ,**surprise** ,**disgust**  &  **neutral**. ")

    elif page == "📷Upload Image":
        st.markdown("<h1 class='title'>📤 Upload</h1>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload an Image",type=["jpg", "png", "jpeg"])

        if uploaded_file:
            image_file = Image.open(uploaded_file)
            st.image(image_file, caption="Uploaded Image", use_container_width=True)


        # Predict emotion
            emotion, confidence = detect_emotion(image_file)

            if emotion:
                st.success(f"🎭 **Predicted Emotion:** {emotion} ({confidence}%)")
                confidence = float(confidence)  # Convert confidence to a Python float
                st.progress(confidence / 100)  # Divide confidence to get a value between 0 and 1
            # Progress bar for confidence
            else:
                st.error(confidence)  # Show error message if prediction fails

# ---- About Page ----
    elif page == "ℹ️About":
        st.markdown("<h1 class='title'>ℹ️ About this App</h1>", unsafe_allow_html=True)
        st.write("    ")
        st.write("This app detects emotions in facial images using a deep learning model.")
        st.write("     ")
        st.write("      ")
        st.write("Developed using 🌐 **Streamlit** , 🧠**TensorFlow** & **Keras**.")
main()


