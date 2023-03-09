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
# Create a title for your app
st.title("Cat vs Dog Image Prediction App")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "gif"])

# Check if a file is uploaded
if uploaded_file is not None:

    # Read and resize the uploaded image
    img_array = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    img_array = cv2.resize(img_array, (224, 224))

    # Display the uploaded image
    st.image(img_array, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image as a temporary file
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict the uploaded image using your function
    prediction, probability = predict_image("temp.jpg")

    # Display the prediction and probability
    st.write(f"Prediction: {prediction}")
    st.write(f"Probability: {probability:.2f}%")
