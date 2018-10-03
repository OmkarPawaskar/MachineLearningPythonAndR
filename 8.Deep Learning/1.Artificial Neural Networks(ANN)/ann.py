#Artificial Neural Network

# Installing Theano - open source numerical computations library- can run on CPU as well as GPU
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing TensorFlow - open source numerical computations library- can run on CPU as well as GPU
# Install TensorFlow from website : https://www.tensorflow.org/versions/r0.11/get_started/
#or go to anaconda prompt - conda create -n tensorflow python = 3.6.6 -> conda activate tensorflow ->pip install --ignore-installed --upgrade tensorflow

# Installing Keras - based on Theano and Tensorflow.
# pip install --upgrade keras

# Part 1 - Data Pre Processing
# Importing the libraries
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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 -Making ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential() #sequence of layers

#3Adding the input layer and first hidden layer

#Number of nodes in input layer is 11 (1 input node = 1 feature for 1 observation)and no. of output node is 1
#therefor to choose number of nodes in hidden layers : 11+1 = 12 /2 ie average =output dim = 6 hidden layers
#input_dim = number of nodes in input layer
#rectifier function for hidden layer , sigmoid function for output layer
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu',input_dim = 11))#relu - rectifier function

#Adding thw Second hidden layer
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu'))

#Adding the output layer
classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))

#Compiling ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics= ['accuracy'])


#Part 3 - Making Predictions and evaluating model
classifier.fit(X_train,y_train,batch_size = 10, nb_epoch = 100) #10 observations at each pass..100 such passes

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #if y_pred > 0.5 it is true if less than false

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)