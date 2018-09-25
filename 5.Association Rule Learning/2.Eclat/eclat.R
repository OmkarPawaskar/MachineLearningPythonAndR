#Eclat - can also say simplified apriori

#Data Preprocessing

#install.packages('arules')
library(arules)
dataset = read.csv("Market_Basket_Optimisation.csv",header = FALSE)
#rm.duplicates = to remove duplicate entries
dataset = read.transactions('Market_Basket_Optimisation.csv',sep = ',',rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset,topN = 100) #to get top 100 products purchased by customers

#Training Eclat on dataset
#support - since we want to optimize rule we need to associate rules for products which are bought atleast 3 times a day 
# 3 * 7 = 21 times week ..min support = 3*7/7500 = 0.0028 = 0.003

rules = eclat(data = dataset , parameter = list(support = 0.004,minlen = 2))

#Visualizing the results
inspect(sort(rules,by = 'support')[1:10]) #sorted first 10 rules by lift

