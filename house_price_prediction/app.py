import streamlit as st
import pickle

model = pickle.load(open('house_price_prediction/trained_model.pkl', 'rb'))

def option_to_num(value):
    if value == 'Yes' or value == 'Semi-Furnished':
        return 1
    elif value == 'Furnished':
        return 2
    else:
        return 0


if __name__ == "__main__":
    st.set_page_config(page_title="House Price Prediction", page_icon="🏠")
    st.title('House Price Prediction')
    area = st.number_input("Area (sq²)", value=0, min_value=0, max_value=None, step=1)
    bedrooms = st.number_input("Bedrooms", value=0, min_value=0, max_value=None, step=1)
    bathrooms = st.number_input("Bathrooms", value=0, min_value=0, max_value=None, step=1)
    stories = st.number_input("Stories", value=0, min_value=0, max_value=None, step=1)
    main_road = option_to_num(st.selectbox("Main Road", ['Yes', 'NO']))
    guestroom = option_to_num(st.selectbox("Guestroom", ['Yes', 'NO']))
    basement = option_to_num(st.selectbox("Basement", ['Yes', 'NO']))
    hot_water = option_to_num(st.selectbox("Hot Water Heating", ['Yes', 'NO']))
    ac = option_to_num(st.selectbox("Air Conditioning", ['Yes', 'NO']))
    parking = st.number_input("Parking", value=0, min_value=0, max_value=None, step=1)
    prefarea = option_to_num(st.selectbox("Prefarea", ['Yes', 'No']))
    furnished_status = option_to_num(st.selectbox("Furnished Status", ['Furnished', 'Semi-Furnished', 'Unfurnished']))

    if st.button("Predict Price"):
        final_input = [[area, bedrooms, bathrooms, stories, main_road, guestroom, basement, hot_water, ac, parking, prefarea, furnished_status]]
        prediction = model.predict(final_input)
        prediction = round(prediction[0], 2)
        st.write(f'The house you are looking for would cost around ₹{prediction[0]}')
