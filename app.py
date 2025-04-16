import streamlit as st
import joblib
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(r"C:\Users\ASUS\Dropbox\PC\Desktop\AI\Heart-Disease-Prediction-using-ML-Algorithms-main\heartdiseasepredivtiondata-firebase-adminsdk-fbsvc-29a700666c.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://heartdiseasepredivtiondata-default-rtdb.firebaseio.com/'
    })

# Load the saved model
try:
    model = joblib.load(r'C:\Users\ASUS\Dropbox\PC\Desktop\AI\Heart-Disease-Prediction-using-ML-Algorithms-main\best_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient information to predict the likelihood of heart disease.")

# User input fields
name = st.text_input("Name")
age = st.number_input("Age", min_value=1, max_value=120, step=1)
oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=10.0, step=0.1)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, step=1)
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-4)", [0, 1, 2, 3, 4])

# Predict button
if st.button("Predict"):
    input_data = np.array([[oldpeak, exang, cp, thalach, ca]])
    prediction = model.predict(input_data)[0]

    result = "Yes (1) - Heart Disease" if prediction == 1 else "No (0) - No Heart Disease"

    # Display result
    if prediction == 1:
        st.error(f"⚠️ The model predicts: **{result}**.")
    else:
        st.success(f"✅ The model predicts: **{result}**.")

    # Prepare data to store in Firebase
    user_data = {
        "Name": name,
        "Age": age,
        "Oldpeak": oldpeak,
        "Exang": exang,
        "CP": cp,
        "Thalach": thalach,
        "CA": ca,
        "Prediction": result,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Push data to Firebase
    try:
        ref = db.reference("/heart_disease_predictions")
        ref.push(user_data)
        st.success("📝 Data successfully saved to Firebase.")
    except Exception as e:
        st.error(f"Failed to save data to Firebase: {e}")
