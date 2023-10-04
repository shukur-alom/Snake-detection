import streamlit as st
from streamlit_lottie import st_lottie
import json
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt




name = ['natrix', 'coronella', 'vipera']

model = tf.keras.models.load_model('Models/CNN.h5')

def load_lottifiel(filepath:str):
    with open(filepath,'r') as f:return json.load(f)

st_lottie( load_lottifiel("snake.json"),height=440)


uploaded_file = st.file_uploader("Upload an image")
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
    normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (224, 224)), axis=0)
    predictions = model.predict(normalized_image)

    # Create a bar chart
    plt.bar([name[0],name[1],name[2]], [predictions[0][0]*100, predictions[0][1]*100, predictions[0][2]*100] )


    plt.title(f"Result is : {name[np.argmax(predictions)]}")

    # Save the bar chart image
    plt.savefig("chart.png")
    
    st.image(image_bytes)
    st.image('chart.png')