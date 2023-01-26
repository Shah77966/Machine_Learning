import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('C:\\Users\\shahk\Documents\\Shah\\Programming\\Machine_Learning\\python_for_microscopists-master\\other_files\\cells.csv')
print(df)
# plt.xlabel('time')
# plt.ylabel('çells')
# plt.scatter(df.time, df.cells, color='red', marker='+')
# plt.show()

#x is independent (time)
#y is  dependent variable (output)

x_df = df.drop('cells', axis = 1)
# x_df = df[['time']]
# print(x_df)
print(x_df.dtypes)
y_df = df[['cells']]

model = linear_model.LinearRegression()   #Created an instance 0f the model
model.fit(x_df, y_df) #Trainig the model (fitting a line)   

print(model.score(x_df, y_df))  #1 means model fits good
print("Predicted number of cells ", model.predict([[2.3]]))

#Y = mx + c
c = model.intercept_
m = model.coef_
print(m*2.3 + c)

cells_pred_df = pd.read_csv('C:\\Users\\shahk\\Documents\\Shah\\Programming\\Machine_Learning\\python_for_microscopists-master\\other_files\\cells_predict.csv')
print(cells_pred_df.head())

predict_cells = model.predict(cells_pred_df)
print(predict_cells)
cells_pred_df['cells'] = predict_cells
print(cells_pred_df.head())
cells_pred_df.to_csv('C:\\Users\\shahk\\Documents\\Shah\\Programming\\Machine_Learning\\python_for_microscopists-master\\other_files\\cells_predicted.csv')