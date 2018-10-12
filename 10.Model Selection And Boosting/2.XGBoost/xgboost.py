
# XGBoost
#Doing ann.py example but faster

#Install XGBoost first 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder #LabelEncoder to encode values and OneHotEncoder to give dummy values
labelEncoder_X_1 = LabelEncoder()
X[:,1]=labelEncoder_X_1.fit_transform(X[:,1]) #to categorize Country column gives encoded values
labelEncoder_X_2 = LabelEncoder()
X[:,2]=labelEncoder_X_2.fit_transform(X[:,2]) #to categorize Country column gives encoded values
onehotencoder = OneHotEncoder(categorical_features=[1]) # to give which column to categorize
X = onehotencoder.fit_transform(X).toarray()
#to remove dummy variable trap
X = X[:,1:]


# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split-it was changed it later version of spyder
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Since we are using XGBoost we wont neeed Feature Scaling(Can keep interpretation of model and problem )

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean() 
accuracies.std()