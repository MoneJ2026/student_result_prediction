import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import joblib


# Load data
data = pd.read_csv("data/students.csv")


# Features
X = data[
    [
        "Math",
        "Programming",
        "Database",
        "Attendance"
    ]
]


# Target
y = data["Average"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = LinearRegression()


# Train model
model.fit(
    X_train,
    y_train
)


# Prediction
prediction = model.predict(X_test)


# Evaluation
print("MAE:", mean_absolute_error(y_test, prediction))

print("R2 Score:", r2_score(y_test, prediction))


# Save model
joblib.dump(
    model,
    "models/student_model.pkl"
)


print("Model saved successfully!")