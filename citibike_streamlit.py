import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Usertype Classifier",
    page_icon="https://account.citibikenyc.com/favicon.ico",
    menu_items={
        "Get help": "mailto:salimkilinc@yahoo.com",
        "About": "For More Information\n" + "https://github.com/salimkilinc"
    }
)

st.title("Usertype Classification Project")

st.markdown("citibike® wants to decide whether the usertype is **:red[Subscriber]** or **:red[Customer]** by looking at the trip data which they get from their database.")

st.image("https://images.ctfassets.net/p6ae3zqfb1e3/5sCUvZowImhNmegssvPVB4/6366a936ae3c82fb5e58843aa339c7bc/Citi_Bike_Plans_pricing_Memberonly_2x.png?w=1500&q=60&fm=webp")

st.markdown("Following recent advancements in the artificial intelligence sector, they anticipate us to create a **machine learning model** that aligns with their requirements and supports their research endeavors.")
st.markdown("Furthermore, upon receiving details about a new user, they desire us to devise a product capable of predicting whether this user is a subscriber or a customer using the provided information.")
st.markdown("*Let's lend our assistance to them!*")

st.image("https://images.ctfassets.net/p6ae3zqfb1e3/5nPSC2OHuwUMCnqkc8XtEe/9cd6f0e16ce73b55579e7de8af2875c9/Citi_Bike_Reduced_fares_Renewing_your_membership_2x.png?w=1500&q=60&fm=webp")

st.header("Data Dictionary")

st.markdown("- **user_type**: Whether a user is a subscriber or a customer (0 = Subscriber, 1 = Customer)")
st.markdown("- **trip_duration**: The length of time in minutes the user spends using the bicycle in a single instance.")
st.markdown("- **start_station_id**: The unique identification number for the station where the user picked up the bicycle.")
st.markdown("- **end_station_id**: The unique identification number for the station where the user left the bicycle")
st.markdown("- **age**: The age of the user.")
st.markdown("- **gender**: The gender of the user. (0 = Male, 1 = Female)")
st.markdown("- **start_hour**: The hour during the day when the user picked up the bicycle. (Time expressed in a 24-hour format.)")
st.markdown("- **start_day_of_week**: The day of the week on which the user picked up the bicycle. (0 = Monday, 1 = Tuesday, 2 = Wednesday, 3 = Thursday, 4 = Friday, 5 = Saturday, 6 = Sunday)")
st.markdown("- **distance**: The distance in meters covered by the user while cycling.")

df = pd.read_csv("citibike_tripdata.csv")
sample_df = df
sample_df['trip_duration'] = sample_df['trip_duration'] / 60
sample_df['trip_duration'] = sample_df['trip_duration'].astype(int)

st.table(sample_df.sample(5, random_state=42))


st.sidebar.markdown("**Select** the features from the options below to view the outcome!")

name = st.sidebar.text_input("Name", help="Please ensure that the initial letter of your name is capitalized.")

surname = st.sidebar.text_input("Surname", help="Please ensure that the initial letter of your surname is capitalized.")

trip_duration_mins = st.sidebar.slider("Trip Duration (min)", min_value=1, max_value=5000)
trip_duration = trip_duration_mins * 60

start_station_id = st.sidebar.slider("Start Station ID", min_value=3000, max_value=4000)

end_station_id = st.sidebar.slider("End Station ID", min_value=3000, max_value=4000)

age = st.sidebar.slider("Age", min_value=0, max_value=100)

def get_gender_input():
    gender = st.sidebar.radio("Gender", ["Male", "Female"])
    return 0 if gender == "Male" else 1

gender = get_gender_input()

start_hour = st.sidebar.slider("Start Hour", min_value=0, max_value=23)

start_day_of_week = st.sidebar.slider("Start Day of Week", min_value=0, max_value=6)

distance = st.sidebar.slider("Distance", min_value=0, max_value=5000)


from joblib import load

rf_model = load('rf_model.pkl')

input_df = pd.DataFrame({
    'trip_duration': [trip_duration],
    'start_station_id': [start_station_id],
    'end_station_id': [end_station_id],
    'age': [age],
    'gender': [gender],
    'start_hour': [start_hour],
    'start_day_of_week': [start_hour],
    'distance': [start_hour]
})

threshold = 0.1

pred_probability = np.round(rf_model.predict_proba(input_df.values)[:, 1], 2)
pred = (pred_probability <= threshold).astype(int)


st.header("Outcome")

# Sonuç Ekranı
if st.sidebar.button("Submit"):

    # Info mesajı oluşturma
    st.info("The outcome is located beneath.")

    # Sorgulama zamanına ilişkin bilgileri elde etme
    from datetime import date, datetime

    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")

    # Sonuçları Görüntülemek için DataFrame
    results_df = pd.DataFrame({
    'Name': [name],
    'Surname': [surname],
    'Date': [today],
    'Time': [time],
    'trip_duration': [trip_duration],
    'start_station_id': [start_station_id],
    'end_station_id': [end_station_id],
    'age': [age],
    'gender': [gender],
    'start_hour': [start_hour],
    'start_day_of_week': [start_day_of_week],
    'distance': [distance],
    'Prediction': [pred]
    })

    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("0","Subscriber"))
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("1","Customer"))

    st.table(results_df)

    if pred == 0:
        st.image("https://images.ctfassets.net/p6ae3zqfb1e3/7ndbyfpZJn84pkh9vgt6aE/aa3a4b7630afdbddd15d8a60055103e1/Citi_Bike_Plans_pricing_Short_term_3x.png?w=1500&q=60&fm=webp")
    else:
        st.image("https://images.ctfassets.net/p6ae3zqfb1e3/6ORxV6gPBOL4ANWPjysVwD/7a416329a33c79950d8a2fc1b960372e/Citi_Bike_Ride_experience_Hero_3x.jpg?w=2500&q=60&fm=webp")
else:
    st.markdown("Please click on **Submit** button!")

