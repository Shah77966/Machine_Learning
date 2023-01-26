import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_excel('C:\\Users\\shahk\\Documents\\Shah\\Programming\\Machine_Learning\\python_for_microscopists-master\\other_files\\images_analyzed.xlsx')

from sklearn import linear_model

model = linear_model.LinearRegression()

model.fit(df[['Time', 'Coffee', 'Age']], df.Images_Analyzed)

prediction = model.predict([[1.4, 0.3, 12]])
print(prediction)