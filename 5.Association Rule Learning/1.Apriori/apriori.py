#Apriori 

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header = None)
#Apriori model needs data as transactions in form of list ..so we will contain list which contains transaction 
#where each transaction is list itself containing set of strings. 
transactions = []
for i in range(0,7501):#for 7501 transcactions ie rows
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])#for 20 columns  
    
#Training Apriori in dataset
from apyori import apriori
#for min support ,consider products bought 3 times a day,means 3*7 times week ..hence min support = 3*7/7500= 0028 or 0.003
#value of lift greater than 3 gives good rules
rules = apriori(transactions , min_support = 0.003 , min_confidence = 0.2 , min_lift= 3 , min_length = 2)


#Visualizing the Results
results = list(rules)