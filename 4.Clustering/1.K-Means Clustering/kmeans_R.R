#K-Means Clustering

#Importing the dataset
dataset <- read.csv('Mall_Customers.csv')
X <- dataset[4:5] #since in R index starts from 1

#Using elbow method to find optimal number of clusters
set.seed(6)
wcss <- vector()
for(i in 1:10) wcss[i]<-sum(kmeans(X,i)$withinss) # to find sum of squared errors of datapoints for different clusters
#type -> p(points),l(lines),b(both)
plot(1:10,wcss,type = 'b',main = paste('Clusters of Clients'),xlab = 'Number of clusters',ylab = 'WCSS')

#Applying Kmeans to mall dataset 
set.seed(29)
#dataset,number of clusters,max_iterations,no. of initial random sets
kmeans <- kmeans(X,5,iter.max = 300,nstart = 10)
y_kmeans = kmeans$cluster

#Visualizing the Clusters
library(cluster)
#check parameters to understand - F1
clusplot(X,
         y_kmeans,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of Clients'),
         xlab = 'Annual Income',
         ylab = 'Spending Scores'
         )