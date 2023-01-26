import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('C:\\Users\\shahk\Documents\\Shah\\Programming\\Machine_Learning\\python_for_microscopists-master\\other_files\\cells.csv')
print(df)

x_df = df.drop('cells', axis=1)
y_df = df.cells

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.4, random_state=10)
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
print(X_train)
print("===========")
print(y_train)

prediction = model.predict(X_test)
print(y_test, prediction)
print("MSE ", np.mean(prediction - y_test)**2)
plt.scatter(prediction, prediction - y_test)
plt.show()