#Natural Language Processing


#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#we will use tsv file cuz reviews and positive/negative mark(1/0) are seperated by tab.
#diff between csv and tsv file is that review and mark as seperated by ' , ' in csv
#user may contain ',' in its review and it may confuse model but they wont press tab and even if they 
#did ,it will move them to next column in form hence tsv file is used.

#Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t',quoting = 3) #'\t' for tab , quoting = 3 for ignoring " ".

#Cleaning the Texts
import re #library which contains cleaning tools
import nltk #contains tools for NLP
nltk.download('stopwords') #stopwords = irrelevant words for eg , and , is .
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #for stemming

corpus = []
for i in range(0,1000):
    #removing all characters except letters
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i]) #sub(characters we want to keep, removed character replaced by space, string to work on)
    review = review.lower()
    review = review.split() #Word Segmentation - splits the sentence string in list of string of words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #to remove stopwords from list.Placed it as set since set of stopwords as it is much faster to run ,than list of stopwords 
    #Applied Lemmatization or Stemming - love,loved,loving are to be taken as same thing by removing suffixes
    
    #Now to get back review string after cleaning
    review = ' '.join(review)
    corpus.append(review)

#Creating Bag of Words Model 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1500) #max_features - to filter out irrelevant words ,Reduces Sparsity, 1500 since there are total 1565 features..its upto us
X = cv.fit_transform(corpus).toarray()
#thus we created BoW
y = dataset.iloc[:, 1].values

#since it is problem of Classification
#Most common models used for Natural Language Provessing is  Decision tree and Naive Bayes
# we will use Naive Bayes

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB() #Naive Bayes has no parameters
classifier.fit(X_train,y_train)

# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy',random_state= 0) #to base it on information gain
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#for Naive Bayes:
#cm = [55, 42],
#       [12, 91]]
#Accuracy = 55+91/200 =0.75 which is good enough
#TN = 55 ,TP = 91 , FN = 12 FP = 42
#precision = TP/(TP+FP) = 91/91+42 =0.68
#Recall = TP/(TP+FN) = 91/91+12 = 0.88
#F1 Score = 2*Precision*Recall / (Precision + Recall) = 0.77

#for Decision Tree:
#cm = [74, 23],
#       [35, 68]
#Accuracy = 74+68/200 =0.72 which is good enough
#TN = 74 ,TP = 68  , FN = 35 FP = 23
#precision = TP/(TP+FP) = 68/68+23 =0.75
#Recall = TP/(TP+FN) = 68/68+35 = 0.66
#F1 Score = 2*Precision*Recall / (Precision + Recall) = 0.86



