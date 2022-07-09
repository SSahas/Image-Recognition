import streamlit as st
import os
import numpy as np
from keras.preprocessing import image
import tensorflow.keras.preprocessing.image
from keras.applications import vgg16

from PIL import Image
st.title('Image Classifier using vgg16 api')


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


#model = pickle.load(open('img_model.p', 'rb'))
model = vgg16.VGG16()
uploaded_file = st.file_uploader("Upload an image....", type=[
                                 "jpg", "png", "svg", "webp"])


if uploaded_file is not None:
    img = Image.open(uploaded_file)
    file_name = uploaded_file.name
    file_details = {"Filename": uploaded_file.name,
                    "FileType": uploaded_file.type, "FileSize": uploaded_file.size}

    imga = load_image(uploaded_file)
    st.image(imga, width=800)
    #st.image(imga, width=800, height = 800)
    with open(file_name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    #imgk =  image.load_img(file_name , target_size=(224, 224))
    imgk = tensorflow.keras.utils.load_img(file_name, target_size=(224, 224))

    if st.button('PREDICT'):
        x = tensorflow.keras.preprocessing.image.img_to_array(imgk)
        x = np.expand_dims(x, axis=0)

        x = vgg16.preprocess_input(x)

        predictions = model.predict(x)

        predicted_classes = vgg16.decode_predictions(predictions, top=9)

        st.write("Top predictions for this image:")

        for imagenet_id, name, likelihood in predicted_classes[0]:
            st.write("Prediction: {} - {:2f}".format(name, likelihood))

        os.remove(file_name)
