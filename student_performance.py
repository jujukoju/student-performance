# Import libraries
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# loading training and testing datasets
train = pd.read_csv("training.csv")
test = pd.read_csv("testing.csv")

# feature selection & feature
features = ["Age", "Gender", "StudyTimeWeekly", "Absences", "Tutoring"]
train = train[features + ['GPA']]
train['GPA'] = train['GPA'].fillna(train['GPA'].median())

X = train[features]
Y = train["GPA"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

rgr = tree.DecisionTreeRegressor()
rgr = rgr.fit(X_train, Y_train)


def get_user_input():
    age = float(input("What's the age?: "))
    gender = int(input("What gender? (0=Female, 1=Male): "))
    study_time = float(input("How many hours per week for study?: "))
    absences = int(input("Total number of absences: "))
    tutoring = int(input("Does student receive tutoring? (0=No, 1=Yes): "))

    return pd.DataFrame({'Age': [age], 'Gender': [gender], 'StudyTimeWeekly': [study_time], 'Absences': [absences],
                         'Tutoring': [tutoring]})


sample_input = get_user_input()

prediction = rgr.predict(sample_input)

print(f"\nThe student's predicted GPA is: {prediction[0]}")

y_pred = rgr.predict(X_test)
mae = mean_absolute_error(Y_test, y_pred)
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")