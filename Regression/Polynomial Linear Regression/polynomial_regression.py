#Polynomial Regression 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #no need to include name of levels #1:2 instead of 1 to make sure we get matrix(10,1) and not array (10,)
y = dataset.iloc[:, 2].values

#Here we need all the levels in training set hence we wont divide it into training and test set
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Linear Regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
#now we have to fit polynomial features in multiple linear model
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualizing the Linear Regression Results
plt.scatter(X, y, color = 'red')
plt.plot(X , lin_reg.predict(X),color = 'blue')
plt.title('Truth or Bluff(Linear Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial Regression Results
#X_grid = np.arange(min(X),max(X),0.1) #divides dataset 1-10 to 1,1.1,1.2,.......,9.9,10.0 for more accurate plot
#X_grid = X_grid.reshape(len(X_grid),1) #after this just replace X by X_grid in plot method
plt.scatter(X, y, color = 'red')
plt.plot(X , lin_reg2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title('Truth or Bluff(Polynomial Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Predicting new result with Linear Regression
lin_reg.predict(6.5)

#Predicting new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))