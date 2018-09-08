# SVR


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values #had to convert array in matrix by [:,2:] instead of [:,2] for feature scaling to work on y

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

#Although previous model handled feature scaling , in SVR model we need to feature scale cause it doesnt do it itself and will give wrong results
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# Fitting the Regression Model to the dataset
# Create your regressor here
from sklearn.svm import SVR
#SVR parameters : C - penalty - used to prevent overfitting(too high C value) and underfitting(too low C value)
#Epsilon -  The value of epsilon determines the level of accuracy of the approximated function
#kernel - specifies the kernel type - linear,poly, rbf,sigmoid.default value is rbf
#rbf = radial basis function
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)


# Predicting a new result
#since X and y are feature scaled manually we have to feature scale this manually too
#Now this gives y_predict in 0.0..... so we have to transform it inversely to get real value
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]])))) #sc takes only array and not datatype hence transform it by numpy
#two pairs of brackets indicates it is array ..if we put only 1 bracket it will consider it as vector since it is ony 1 element

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#Here CEO point is not considered in SVR line since it is considered as outlier by SVM


# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()