# Simple Linear Regression 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # till second last column
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training Set

# we didnt feature scaled here because the library we are gonna choose will take care of that for us

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)

#Visualizing the Training Set Results
#scatter draws points without lines connecting them whereas plot may or may not plot the lines, 
#depending on the arguments.
plt.scatter(X_train, y_train, color = 'red') #gives observation points
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #gives regression line
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the Test Set Results

plt.scatter(X_test, y_test, color = 'red') #gives observation points
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #we have already test it above so we can use that model here too .
#since we are just seeing whether line predicted in training set matches test set as well.
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
