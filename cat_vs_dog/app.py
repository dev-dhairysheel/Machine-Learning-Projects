import streamlit as st
import numpy as np
import cv2
import pickle
import tempfile
from PIL import Image

# Load the pre-trained model
model = pickle.load(open('cat_vs_dog/cats_vs_dogs.pkl', 'rb'))

# Define the predict_image function to make predictions on new images
def predict_image(uploaded_file):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        imagepath = tmp_file.name

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
    return prediction

# Set the title and the sidebar heading of the app
st.title("Cat or Dog Classifier")
st.sidebar.title("Upload Image")

# Create the file uploader to upload new images
uploaded_file = st.sidebar.file_uploader(label="Upload an image", type=["jpg", "jpeg", "png"])

# Check if an image has been uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)

    # Make a prediction on the uploaded image
    prediction = predict_image(uploaded_file)
    if prediction == "none":
        st.warning("Sorry, we could not make a prediction on this image.")
    else:
        st.success(f"The uploaded image is a {prediction}.")
