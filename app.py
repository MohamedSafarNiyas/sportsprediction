

import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from PIL import Image
import numpy as np
st.title("Sports Image Classification")

st.write("Predict the sport that is being represented in the image.")

model = load_model("sports.h5",custom_objects={'KerasLayer':hub.KerasLayer})
labels = {
        0: 'air hockey',
        1: 'boxing',
        2: 'cricket',
        3: 'football',
        4: 'formula 1 racing',
        5: 'luge',
        6: 'motorcycle racing'
    }




uploaded_file = st.file_uploader(
    "Upload an image of a sport being played:", type='jpg'
)
predictions=-1
if uploaded_file is not None:
    image1 = Image.open(uploaded_file)
    image1=image.smart_resize(image1,(224,224))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = model.predict(img_array)
    label=labels[np.argmax(predictions)]


st.write("### Prediction Result")
if st.button("Predict"):
    if uploaded_file is not None:
        image1 = Image.open(uploaded_file)
        st.image(image1, caption="Uploaded Image", use_column_width=True)
        st.markdown(
            f"<h2 style='text-align: center;'>Image of {label}</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.write("Please upload file or choose sample image.")


st.write("If you would not like to upload an image, you can use the sample image instead:")
sample_img_choice = st.button("Use Sample Image")

if sample_img_choice:
    image1 = Image.open("cricket.jpg")
    image1=image.smart_resize(image1,(224,224))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = model.predict(img_array)
    label=labels[np.argmax(predictions)]
    image1 = Image.open("cricket.jpg")
    st.image(image1, caption="Uploaded Image", use_column_width=True)    
    st.markdown(
        f"<h2 style='text-align: center;'>{label}</h2>",
        unsafe_allow_html=True,
    )
