import joblib
import pandas as pd


# Load trained model
model = joblib.load("models/student_model.pkl")


print("🎓 Student Result Prediction")


# User input
math = float(input("Math score: "))
programming = float(input("Programming score: "))
database = float(input("Database score: "))
attendance = float(input("Attendance (%): "))


# Create DataFrame
input_data = pd.DataFrame(
    [[math, programming, database, attendance]],
    columns=[
        "Math",
        "Programming",
        "Database",
        "Attendance"
    ]
)


# Prediction
prediction = model.predict(input_data)


# Output
print("------------------------")
print("Predicted Average Score:")
print(round(prediction[0], 2))


# Grade
score = prediction[0]

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


print("Predicted Grade:")
print(grade)