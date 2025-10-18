import pandas as pd
import numpy as np

data = pd.read_csv("C:/Users/diego/JetLearn/ML & AI/Lesson 12/loan_approval.csv")

print(data.head(20))
print(data.info())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['loan_approved'] = le.fit_transform(data["loan_approved"])

print(data.info())

y = data['loan_approved']
x = data[['income', 'credit_score', 'loan_amount', 'years_employed', 'points']]

print(data.info())

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(x_train, y_train)

from sklearn.metrics import mean_squared_error

ypredict = linreg.predict(x_test)

rmse_linear = np.sqrt(mean_squared_error(y_test, ypredict))

print("The RMSE is case of multivariable regression is", rmse_linear)

from sklearn.preprocessing import PolynomialFeatures

poly_feature = PolynomialFeatures(degree = 2)

x_train_poly = poly_feature.fit_transform(x_train)
x_test_poly = poly_feature.fit_transform(x_test)

polyreg = LinearRegression()
polyreg.fit(x_train_poly, y_train)

ypredict_poly = polyreg.predict(x_test_poly)

rmse_poly = np.sqrt(mean_squared_error(y_test, ypredict_poly))

print("The RMSE is case of polynomial regression is", rmse_poly)
