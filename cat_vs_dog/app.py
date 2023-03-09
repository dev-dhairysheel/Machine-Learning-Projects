import pickle
import cv2
import numpy as np
import streamlit as st

model = pickle.load(open('cats_vs_dogs.pkl', 'rb'))

#to predict new images 
def predict_image(imagepath):
    img_array = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
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


st.set_page_config(page_title='Cats vs Dogs')
upload_img = st.file_uploader('Please choose an image', type=['png', 'jpg', 'jped'], accept_multiple_files=False)
st.image(upload_img)
predict_image(upload_img)
st.write(f'It is is {prediction}')
