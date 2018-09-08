# Data Preprocessing

# Importing the libraries

import numpy as np #contains mathematical tools 
import matplotlib.pyplot as plt #plot charts
import pandas as pd #to import and manage datasets

# Importing dataset
dataset = pd.read_csv('Data.csv') #reading dataset
# iloc -> integer-location based indexing for selection by position.
X = dataset.iloc[:, :-1].values #taking all columns except last one which is output label
Y = dataset.iloc[:, 3].values #taking column of output label 

# Taking care of missing data
from sklearn.preprocessing import Imputer #for completing missing values .Select and Press Ctrl +I to see syntax
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)#Replace missing values by mean
imputer = imputer.fit(X[:,1:3]) #since index 1 and 2 contains missing columns
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder #LabelEncoder to encode values and OneHotEncoder to give dummy values
labelEncoder_X = LabelEncoder()
X[:,0]=labelEncoder_X.fit_transform(X[:,0]) #to categorize Country column gives encoded values
onehotencoder = OneHotEncoder(categorical_features=[0]) # to give which column to categorize
X = onehotencoder.fit_transform(X).toarray()

#to categorize output label  it  wont need OneHotEncoder since it is dependent variable with only 2 labels Yes or No
labelEncoder_Y = LabelEncoder()
Y=labelEncoder_Y.fit_transform(Y) #to categorize Purchased column gives encoded values

# Splitting the data into training set and test set
from sklearn.cross_validation import train_test_split
#random_state simply sets a seed to the random generator, so that your train-test splits are always deterministic. If you don't set a seed, it is different each time.
#This ensures that the random numbers are generated in the same order
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0) #20% data as test set and 80% training set

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # no need to fit since it is already fitted
