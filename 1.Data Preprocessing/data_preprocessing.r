#Data preprocessing
#no need to import libraries in R as there is already libraries imported in Packages

#Importing dataset

dataset = read.csv('Data.csv')

#Taking care of missing data
#ifelse(if true,then return,dataset)
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x,na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x,na.rm = TRUE)),
                        dataset$Salary)
#Encoding categorical data
dataset$Country = factor(dataset$Country,
                         levels = c('France','Spain','Germany'),
                         labels = c(1,2,3))
dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No','Yes'),
                           labels = c(0,1))

#Splitting dataset into training and test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased,SplitRatio = 0.8)
training_set = subset(dataset,split == TRUE)
test_set = subset(dataset,split == FALSE)

#Feature scaling
training_set[, 2:3] = scale(training_set[, 2:3]) #applying it only on columns which are numeric and doesnt contain strings
test_set[, 2:3] = scale(test_set[, 2:3])
