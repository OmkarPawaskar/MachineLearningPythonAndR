# Logistic Regression
# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[,3:5]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature Scaling
training_set[,1:2] = scale(training_set[,1:2])
test_set[1:2] = scale(test_set[,1:2])

#Fitting Logistic Regression to Training Set
classifier= glm(formula = Purchased ~ .,
                family = binomial,#for logistic regression
                data = training_set)

#Predicting the Test Set results
prob_pred = predict(classifier,type = 'response', newdata = test_set[-3])#removed last column ie dependent variable from test set
#Now to get result in 0 or 1 rather than continuous probabilities
y_pred = ifelse(prob_pred > 0.5,1,0)

#Making Confusion Matrix
cm = table(test_set[,3],y_pred)
# cm =  y_pred
#0  1
#0 57  7
#1 10 26
#means 57+26 = correct predictions and 10 + 7 = 17 incorrect predictions

#Visualizing the Training Set results
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_pred = predict(classifier,type = 'response', newdata = test_set[-3])
y_pred = ifelse(prob_pred > 0.5,1,0)
plot(set[, -3],
     main = 'Classifier (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_pred = predict(classifier,type = 'response', newdata = test_set[-3])
y_pred = ifelse(prob_pred > 0.5,1,0)
plot(set[, -3], main = 'Classifier (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))