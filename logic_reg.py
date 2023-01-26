import pandas as pd
from matplotlib import pyplot as plt
#Step 1: Data reading
df = pd.read_csv('C:\\Users\\shahk\\Documents\\Shah\\Programming\\Machine_Learning\\python_for_microscopists-master\\other_files\\images_analyzed_productivity1.csv')

#Step 2: Data Understanding

# print(df.head())
plt.scatter(df.Time, df.Productivity, marker='+', color='red')
# plt.show()
sizes = df['Productivity'].value_counts(sort=1)
plt.pie(sizes, autopct='%.2f%%')
# plt.show()

#Step 3: Drop Irrelevent Data (Data cleaning)

df.drop(['Images_Analyzed'], axis=1, inplace=True)
df.drop(['User'], axis=1, inplace=True)
# print(df.head())

#Step 4: Deal with missing values

df.dropna()

#Step 5: Convert objects to numbers

df.Productivity[df.Productivity == 'Good'] = 1
df.Productivity[df.Productivity == 'Bad'] = 0
# print(df.head())

#Step 6: Prepare the data (Define independent / independent variables)

Y = df['Productivity'].values
# print(Y.dtype)      #object
Y = Y.astype('int')     #convert object to integer
# print(Y.dtype)      #int

X = df.drop(labels = ['Productivity'], axis=1)
# print(X.head())

#Step 7: Split data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = (train_test_split(X, Y, test_size=0.1, random_state=20))

#Step 8: Defint the model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()    #create an instance
model.fit(X_train, y_train)

#Step 9: Testing the model
prediction_test = model.predict(X_test)

from sklearn import metrics
print("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))

#Step 10: Weights
# print(model.coef_)
weights = pd.Series(model.coef_[0], index=X.columns.values)
print(weights)