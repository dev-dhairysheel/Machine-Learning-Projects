from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('cats_vs_dogs/cats_vs_dogs.h5')

# Define a function to predict the uploaded image
def predict_image(image):
    # Convert the image to grayscale and resize to 60x60
    image = image.convert('L')
    image = image.resize((60, 60))

    # Convert the image to a numpy array and normalize it
    image = np.array(image) / 255.0
    image = image.reshape((-1, 60, 60, 1))

    # Predict the class of the image
    result = model.predict(image)
    if result[0][0] >= 0.5:
        prediction = f'dog'
        probability = (result[0][0]) * 100
        if probability < 90:
            prediction = 'none'
    else:
        prediction = 'cat'
        probability = 1 - result[0][0]
        probability = probability * 100
        if probability < 90:
            prediction = 'none'

    return prediction, probability

# Set the title of the app
st.set_page_config(page_title='Cats vs Dogs', page_icon=':dog:')

# Create a sidebar for instructions
st.sidebar.title("Instructions")
st.sidebar.write("1. Upload an image of a cat or dog (in JPG, PNG, or JPEG format).")
st.sidebar.write("2. Wait for the image to be uploaded and the prediction to be made.")
st.sidebar.write("3. View the prediction result and the confidence level.")

# Create a title for your app
st.title("Cat vs Dog Image Prediction App")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

# Check if a file is uploaded
if uploaded_file is not None:

    # Display a spinner while the image is being uploaded and the prediction is being made
    with st.spinner("Predicting the contents of the uploaded image..."):
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict the uploaded image using your function
        prediction, probability = predict_image(image)

        # Display the prediction and probability
        if prediction == "none":
            st.markdown("**Unable to make a prediction**: Our model has detected no cats or dogs in the uploaded image. Please try again with a different image if you are looking for cats or dogs.")
        else:
            st.markdown(f"**Prediction result:** Our algorithm has determined that this image contains a **{prediction}** with **{probability:.2f}%** confidence.")

        st.error('Notice: Our image recognition model is a statistical algorithm that provides an estimate of the likelihood of an image containing a cat or dog. While our model has a high degree of accuracy, it may occasionally provide false results. Please use this tool with caution and always verify the results.')
