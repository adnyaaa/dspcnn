import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# Labels
labels_2_class = ['Healthy', 'Parkinson']
labels_3_class = ['Healthy', 'Parkinson', 'Random']

# Load models
model_2_class = load_model('parkinson_modelnormal957.h5')
model_3_class = load_model('withrandom.h5')

# Streamlit UI
st.title('Use the Model for Prediction')
st.markdown('<h4 style="color: black;">Use the Sidebar to upload an image and predict the class.</h4>', unsafe_allow_html=True)

st.sidebar.markdown('<h2 style="color: blue;">Upload an image to predict the class</h2>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png"])

# Add a slider for threshold adjustment
threshold = st.sidebar.slider('Set threshold for 2-Class Model', 0.0, 1.0, 0.8)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the image to RGB if it has an alpha channel (RGBA)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Preprocess the image
    img = image.resize((350, 350))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    st.write("Image array shape:", x.shape)  # Debug output

    # Prediction for model 2 class
    try:
        prediction_2_class = model_2_class.predict(x)
        st.write("2-Class Model raw prediction:", prediction_2_class)  # Debug output
        predicted_class_2_class = labels_2_class[np.argmax(prediction_2_class)]
        highest_prob_2_class = np.max(prediction_2_class)

        # Apply threshold logic for 2-class model
        if highest_prob_2_class < threshold:
            predicted_class_2_class = 'Random'
    except Exception as e:
        st.write("Error in 2-Class Model prediction:", e)
        predicted_class_2_class = 'Error'

    # Prediction for model 3 class with 'Random'
    try:
        prediction_3_class = model_3_class.predict(x)
        st.write("3-Class Model (random) raw prediction:", prediction_3_class)  # Debug output
        predicted_class_3_class = labels_3_class[np.argmax(prediction_3_class)]
    except Exception as e:
        st.write("Error in 3-Class Model (random) prediction:", e)
        predicted_class_3_class = 'Error'

    # Display results for 2-Class Model
    st.subheader('2-Class Model Prediction')
    st.write(f'The predicted class is: {predicted_class_2_class}')

    if predicted_class_2_class != 'Error':
        st.subheader('2-Class Model Predicted Output Chart')
        probabilities_2_class = prediction_2_class[0]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=labels_2_class + ['Random'], y=np.append(probabilities_2_class, 1 - highest_prob_2_class), palette='viridis')
        ax.set_title('Predicted Class Probabilities')
        ax.set_ylabel('Probability')
        plt.tight_layout()
        st.pyplot(fig)

    # Display results for 3-Class Model with 'Random'
    st.subheader('3-Class Model Prediction (random)')
    st.write(f'The predicted class is: {predicted_class_3_class}')

    if predicted_class_3_class != 'Error':
        st.subheader('3-Class Model Predicted Output Chart (random)')
        probabilities_3_class = prediction_3_class[0]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=labels_3_class, y=probabilities_3_class, palette='viridis')
        ax.set_title('Predicted Class Probabilities')
        ax.set_ylabel('Probability')
        plt.tight_layout()
        st.pyplot(fig)
else:
    st.write("Please upload an image to predict.")
