#Multiple linear Regression

# Importing the dataset
dataset = read.csv('50_Startups.csv')

#Encoding categorical data
#factor her encodes and creates dummy variables itself laters
dataset$State = factor(dataset$State,
                         levels = c('New York','California','Florida'),
                         labels = c(1,2,3))


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

#Fitting Multiple Linear Regression to Training Set
#regressor = lm(formula = Profit ~ R.D.Spend + Administration +Marketing.Spend+State,data = training set)
#OR
#here profit is expressed as linear combination of all independent variables by '.'
regressor = lm(formula = Profit ~ . , data= training_set)
#if u check summary(regressor) in console you will see that R.D.Spend is powerful predictor among other features
#hence we can also do 
#regressor = lm(formula = Profit ~ R.D.Spend , data= training_set)
#and we will still get same result

#Predicting the Test Set Results
y_pred = predict(regressor,newdata = test_set)

#Building  the optimal model using Backward Elimination
#this model selects features which are actually significant for better result and hence gives optimized results
#here we do it for whole dataset (though doing it for training set will give same result as well)
#Lower the value the more  your independent variable will have high impact on your dependent variable

#after summary(regressor) check no. of stars after predictor ..that gives significant predictor which satisfies significant level


regressor = lm(formula = Profit ~ R.D.Spend + Administration +Marketing.Spend+State,data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration +Marketing.Spend,data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,data = dataset)
summary(regressor)

#here Marketing.Spend had p value of 0.6 ie 6% which is fine too but just for method purpose lets do this
regressor = lm(formula = Profit ~ R.D.Spend ,data = dataset)
summary(regressor)

