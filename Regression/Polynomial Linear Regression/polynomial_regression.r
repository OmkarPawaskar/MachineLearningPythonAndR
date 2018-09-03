#Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

#Fitting Linear Regression to dataset
lin_reg = lm(formula =Salary ~ . ,data = dataset )

#Fitting Polynomial Regression to dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ . , data = dataset)

#Visualizing Linear Regression Results
#install.packages(ggplot2)
library(ggplot2)
ggplot()+
  geom_point(aes(x =dataset$Level , y = dataset$Salary),
             colour = 'red')+
  geom_line(aes(x =dataset$Level , y = predict(lin_reg,newdata = dataset)),
            colour = 'blue')+
  ggtitle('Truth or Bluff(Linear Regression)')+
  xlab('Level ')+
  ylab('Salary ')

#Visualizing Polynomial Regression Results
library(ggplot2)
ggplot()+
  geom_point(aes(x =dataset$Level , y = dataset$Salary),
             colour = 'red')+
  geom_line(aes(x =dataset$Level , y = predict(poly_reg,newdata = dataset)),
            colour = 'blue')+
  ggtitle('Truth or Bluff(Polynomial Regression)')+
  xlab('Level ')+
  ylab('Salary ')

#Predicting new result with Linear Regression 
y_pred = predict(lin_reg,newdata = data.frame(Level = 6.5)) #in R you need data to be present in dataset to predict hence we add data by dataframe
#By adding 6.5 to Level column

#Predicting new result with Polynomial Regression
y_pred1 = predict(poly_reg,newdata = data.frame(Level = 6.5,
                                               Level2 = 6.5^2,
                                               Level3 = 6.5^3,
                                               Level4 = 6.5^4))