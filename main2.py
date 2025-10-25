# A redo of a previous algorithm
import pandas as pd
import matplotlib.pyplot as matpat
import seaborn as sans

data = pd.read_csv("C:/Users/diego/JetLearn/ML & AI/Lesson 12/Employee.csv")

print(data.head(20))
print(data.info())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["Education"] = le.fit_transform(data["Education"])
data["Gender"] = le.fit_transform(data["Gender"])
data["EverBenched"] = le.fit_transform(data["EverBenched"])

print(data.head())
print(data.info())

x = data[["Education", "JoiningYear", "PaymentTier", "Age", "Gender", "EverBenched", "ExperienceInCurrentDomain"]]
y = data["LeaveOrNot"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)

from sklearn.tree import DecisionTreeClassifier

classify = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classify.fit(x_train, y_train)
y_predict = classify.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score

neo = confusion_matrix(y_test, y_predict)

sans.heatmap(neo, annot = True, fmt = 'd')

matpat.title("Confusion Matrix")
matpat.xlabel("Prediction")
matpat.ylabel("Actuality")
matpat.show()

acc = accuracy_score(y_test, y_predict)
print("accuracy =", round((acc * 100), 2), "%")
