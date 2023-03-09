import pickle
import cv2
import numpy as np
import streamlit as st
from PIL import Image

model = pickle.load(open('cat_vs_dog/cats_vs_dogs.pkl', 'rb'))

#to predict new images 
def predict_image(imagepath):
    img_array = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    print(f"img_array: {img_array}")
    new_array = cv2.resize(img_array, (60, 60))
    input_image = np.array(new_array).reshape(-1, 60, 60, 1) / 255.0
    result = model.predict(input_image)
    result = np.array(result)
    print(result)
    print(result.ndim)
    if result[0][0] >= 0.5:
        prediction = 'dog'
        probability = (result[0][0])*100
        if probability < 90:
            prediction = 'none'
    else:
        prediction = 'cat'
        probability = 1 - result[0][0]
        probability = probability*100
        if probability < 90:
            prediction = 'none'
    return prediction

st.title("Cat vs Dog Classifier")
st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    prediction = predict_image(uploaded_file)
    st.write(f"Prediction: {prediction}")
