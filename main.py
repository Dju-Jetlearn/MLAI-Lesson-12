#A Redo of a previous algorithm

import pandas as pd
import numpy as np

data = pd.read_csv("C:/Users/diego/JetLearn/ML & AI/Lesson 12/loan_approval.csv")

print(data.head(20))
print(data.info())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["city"] = le.fit_transform(data["city"])

data['loan_approved'] = le.fit_transform(data["loan_approved"])

print(data.info())

