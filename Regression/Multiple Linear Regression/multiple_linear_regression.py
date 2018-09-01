#Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder #LabelEncoder to encode values and OneHotEncoder to give dummy values
labelEncoder_X = LabelEncoder()
X[:,3]=labelEncoder_X.fit_transform(X[:,3]) #to categorize Country column gives encoded values
onehotencoder = OneHotEncoder(categorical_features=[3]) # to give which column to categorize
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X = X[:, 1:] #dont need to do this manually,Library usually takes care of this.But just to show how it works

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


#Fitting Multiple Linear Regression to Training Set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting Test Set Results
y_pred = regressor.predict(X_test)


#Buidling the optimal model using Backward Elimination
#this model selects features which are actually significant for better result and hence gives optimized results
import statsmodels.formula.api as sm
#we need to add column of 1s for libarary to understand that it is b0x0+b1x1... else it will take it as b1x1+b2x2...
#adding ones to X or as shown here adding X to column of 1s
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1 ) #axis = 1 for column , = 0 for row
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
#OLS = ordinary least squares
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit() #step 2 : fit full model with all possible predictors..check blueprint for building model in Multiple Regression folder
regressor_OLS.summary() #now here index 2 has P value of 99% ..way bigger than 5% significance level so..we remove it
X_opt = X[:,[0, 1, 3, 4, 5]]#repeat till all Predictors have P value less than significance level ie 5%
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit() 
regressor_OLS.summary()
X_opt = X[:,[0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit() 
regressor_OLS.summary()
X_opt = X[:,[0, 3, 5]] #this has p value 0.06 ie 6% ..it is ok but just to follow procedure of less than 5% remove this as well 
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit() 
regressor_OLS.summary()
X_opt = X[:,[0, 3]] #here x 1 doesnt have p value of 0 but it is so small that it is in 0.000000something
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit() 
regressor_OLS.summary()