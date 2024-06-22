import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from PIL import Image
from io import StringIO
from main import bld
import tensorflow as tf
import load_model

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    interpreter = tf.lite.Interpreter(model_path="model4.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    data = bld(opencv_image)

    gray_image = data.detect()
    crop = data.crp()

    st.image(gray_image)
    
    for i in range(len(crop)):
        st.image(crop[i])
        
    input_shape = input_details[0]['shape']
    input_data = load_model.load_and_preprocess_image(uploaded_file, input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)

    if predicted_class == 0:
        st.markdown('nama penyakit : EarlyPreB')
    elif predicted_class == 1:
        st.markdown('nama penyakit : PreB')
    elif predicted_class == 2:
        st.markdown('nama penyakit : ProB')
    elif predicted_class == 3:
        st.markdown('nama penyakit : Benign')
    else:
        st.markdown('nama penyakit : Normal')