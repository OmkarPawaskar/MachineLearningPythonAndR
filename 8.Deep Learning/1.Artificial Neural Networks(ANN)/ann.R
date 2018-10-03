# Artificial Neural Network

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

# Encoding the categorical variables as factors
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

# Fitting ANN to the Training set
# install.packages('h2o')
#one of best packages for deep learning.
#diff packages such as neuralnet,mnet(only for one hidden layer),deepnet(for many hidden layers)
#3 reasons why h20 is better : open source,faster computation(efficiency),provides lot of options,
#also contains parameter tuning 
library(h2o) 
h2o.init(nthreads = -1) #nthreads = number of cores in system.. -1 specifies using all cores of cpu
model = h2o.deeplearning(y = 'Exited',
                         training_frame = as.h2o(training_set),
                         activation = 'Rectifier',#rectifier function for hidden layers
                         hidden = c(6,6), #(no. of hidden layers,no. of nodes in hidden layer)
                         #5 because avg(indp var(10),dep var(1)) = 5
                         epochs = 100,#noo. of times to be iterated
                         train_samples_per_iteration = -2) #-2 for autotuning

# Predicting the Test set results
y_pred = h2o.predict(model, newdata = as.h2o(test_set[-11]))
y_pred = (y_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)
#accuracy = 1542+180/2000 = 0.861= 86%

h2o.shutdown()