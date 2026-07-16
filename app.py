import streamlit as st
import pandas as pd
import joblib


# Load model
model = joblib.load(
    "models/student_model.pkl"
)


# App title
st.title("🎓 Student Result Prediction System")

st.write(
    "Enter student scores to predict average result"
)


# Inputs

math = st.number_input(
    "Math Score",
    min_value=0,
    max_value=100,
    value=50
)


programming = st.number_input(
    "Programming Score",
    min_value=0,
    max_value=100,
    value=50
)


database = st.number_input(
    "Database Score",
    min_value=0,
    max_value=100,
    value=50
)


attendance = st.number_input(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=80
)


# Button

if st.button("Predict Result"):

    input_data = pd.DataFrame(
        [[
            math,
            programming,
            database,
            attendance
        ]],
        columns=[
            "Math",
            "Programming",
            "Database",
            "Attendance"
        ]
    )


    prediction = model.predict(
        input_data
    )


    score = prediction[0]


    st.success(
        f"Predicted Average Score: {score:.2f}"
    )


    # Grade

    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"


    st.info(
        f"Predicted Grade: {grade}"
    )