# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Required Libraries

2.Load the Dataset

3.Copy Data & Drop Unwanted Columns

4.Check Data Quality

5.Encode Categorical Variables

6.Define Features (X) and Target (y)

7.Split into Training and Testing Sets

8.Build and Train Logistic Regression Model

9.Make Predictions

10.Evaluate the Model

11.Predict for a New Student

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Imesh Karthick P R
RegisterNumber: 212225060092  
*/

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("Placement_Data.csv")
print("First 5 rows of the dataset:")
print(data.head())
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)
print("\nData after dropping 'sl_no' and 'salary':")
print(data1.head())
print("\nChecking for missing values (True = missing):")
print(data1.isnull().any())
print("\nNumber of duplicate rows:")
print(data1.duplicated().sum())
cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", 
            "degree_t", "workex", "specialisation", "status"]
le = LabelEncoder()
for col in cat_cols:
    data1[col] = le.fit_transform(data1[col])
print("\nData after Label Encoding:")
print(data1.head())
X = data1.iloc[:, :-1]
y = data1["status"]

print("\nFeatures (X) sample:")
print(X.head())

print("\nTarget (y) sample:")
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("\nTraining and testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
lr = LogisticRegression(solver="liblinear")
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("\nPredicted values (y_pred):")
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
new_student = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]

new_prediction = lr.predict(new_student)

print("\nPrediction for new student (0 = Not Placed, 1 = Placed):")
print(new_prediction[0])

```

## Output:
<img width="640" height="515" alt="image" src="https://github.com/user-attachments/assets/9764bb61-d0e9-4608-8673-9c5bb311de5b" />
<img width="662" height="462" alt="image" src="https://github.com/user-attachments/assets/9d35844e-af42-47cc-ac07-be1a53384c3c" />
<img width="722" height="622" alt="image" src="https://github.com/user-attachments/assets/39b04a15-4e8a-42a4-b143-5c583cf44f2c" />
<img width="1437" height="372" alt="image" src="https://github.com/user-attachments/assets/4479f4ab-a0fa-46e8-b5ab-ca1db50c9478" />





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
