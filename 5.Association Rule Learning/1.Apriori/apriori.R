#Apriori

#Data Preprocessing
#install.packages('arules')
library(arules)
dataset = read.csv("Market_Basket_Optimisation.csv",header = FALSE)
#rm.duplicates = to remove duplicate entries
dataset = read.transactions('Market_Basket_Optimisation.csv',sep = ',',rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset,topN = 100) #to get top 100 products purchased by customers

#Training Apriori on dataset
#support - since we want to optimize rule we need to associate rules for products which are bought atleast 3 times a day 
# 3 * 7 = 21 times week ..min support = 3*7/7500 = 0.0028 = 0.003
#confidence = we want higher confidence between 2 products. hence 0.8
#BUT we got 0 rules since we need to get 4 out 5 conditions correct which out dataset doesnt provide hence:
#rules = apriori(data = dataset , parameter = list(support = 0.003 ,confidence = 0.8))
#rules = apriori(data = dataset , parameter = list(support = 0.003 ,confidence = 0.4))
#rules = apriori(data = dataset , parameter = list(support = 0.003 ,confidence = 0.2))
# ppl buying item min 4 times a day , 4*7 / 7500 = 0.004
rules = apriori(data = dataset , parameter = list(support = 0.004 ,confidence = 0.2))

#Visualizing the results
inspect(sort(rules,by = 'lift')[1:10]) #sorted first 10 rules by lift
