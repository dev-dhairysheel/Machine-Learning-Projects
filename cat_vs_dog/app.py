import pickle
import cv2
import numpy as np
from PIL import Image
import streamlit as st

model = pickle.load(open('cat_vs_dog/cats_vs_dogs.pkl', 'rb'))

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

# Set the title of the app
st.title("Cat vs. Dog Classifier")

# Create a file uploader widget in Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded
if uploaded_file is not None:
    # Display the image on the app
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Make a prediction and display the result
    prediction = predict_image(uploaded_file)
    if prediction == 'dog':
        st.write("I'm pretty sure that's a dog!")
    elif prediction == 'cat':
        st.write("I'm pretty sure that's a cat!")
    else:
        st.write("I'm not sure what that is...")

