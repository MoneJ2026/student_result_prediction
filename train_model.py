import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import joblib


# ==========================================
# 1. Load Dataset
# ==========================================

data = pd.read_csv("data/students.csv")


# ==========================================
# 2. Features
# ==========================================

X = data[
    [
        "Math",
        "Programming",
        "Database",
        "Attendance"
    ]
]


# ==========================================
# 3. Target
# ==========================================

y = data["Average"]


# ==========================================
# 4. Split Data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 5. Create Model
# ==========================================

model = LinearRegression()


# ==========================================
# 6. Train Model
# ==========================================

model.fit(
    X_train,
    y_train
)


# ==========================================
# 7. Make Predictions
# ==========================================

prediction = model.predict(X_test)


# ==========================================
# 8. Evaluate Model
# ==========================================

mae = mean_absolute_error(
    y_test,
    prediction
)

r2 = r2_score(
    y_test,
    prediction
)


print("================================")
print("Student Result Prediction Model")
print("================================")

print("MAE:", mae)
print("R2 Score:", r2)


# ==========================================
# 9. Save Model
# ==========================================

joblib.dump(
    model,
    "models/student_model.pkl"
)


print("================================")
print("Model saved successfully!")
print("Location: models/student_model.pkl")
print("================================")