# A redo of a previous algorithm

import seaborn as sans
import numpy as np
import matplotlib.pyplot as matpat
import pandas as pd

data = pd.read_csv("C:/Users/diego/JetLearn/ML & AI/Lesson 12/CollegePlacement.csv")

print(data.head())
print(data.info())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["Internship_Experience"] = le.fit_transform(data["Internship_Experience"])
data["Placement"] = le.fit_transform(data["Placement"])

print(data.info())


X = data[["IQ", "Prev_Sem_Result", "CGPA", "Academic_Performance", "Internship_Experience", "Extra_Curricular_Score", "Communication_Skills", "Projects_Completed"]]

Y = data["Placement"]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)

from sklearn.ensemble import RandomForestClassifier

#n_estimators = 100 - num of trees

classifier = RandomForestClassifier(n_estimators = 100)

classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

matrix_neo = confusion_matrix(Y_test, Y_pred)

sans.heatmap(matrix_neo, annot = True, fmt = "d")

matpat.title("Confusion Matrix")
matpat.xlabel("Prediction")
matpat.ylabel("Actuality")
matpat.show()

acc = accuracy_score(Y_test, Y_pred)
print("accuracy =", round((acc * 100), 2), "%")