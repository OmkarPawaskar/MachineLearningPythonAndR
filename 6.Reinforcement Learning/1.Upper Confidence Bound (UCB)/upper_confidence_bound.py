#Upper Confidence Bound

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
import math
N = 10000 #10000 users or rounds
d = 10
ads_selected = [] #list of all different versions of the ad selected at each round.
numbers_of_selections = [0]*d #vector of size d containing 0s #numbers_of_selections is the variable to stores the value of selected item.
sums_of_rewards = [0]*d
total_reward = 0
for n in range(0,N):
    max_upper_bound = 0
    for i in range(0,d):
        if numbers_of_selections[i]>0:
            #UCB 
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            #upper bound confidence = average_reward + delta_i
            delta_i = math.sqrt(3/2 * (math.log(n + 1) / numbers_of_selections[i])) #since n+1 because index in python starts from 0 and in dataset it starts from 1
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400 #10^400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound #to find max upper bound 
            ad = i #to keep track of index 
            #for n = 0,number of selections will never be greater than 0 hence max_upper_bound = 1e400 and ad = 0 . since 
            #for any i value ie 1,2,3... it will be 1e400 hence upper_bound > max_upper_bound will not be true hence ad will always be 0(for n = 0).
            
    ads_selected.append(ad)       
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1 #total how many times particular ad got clicked.
    reward = dataset.values[n,ad] #n = index of row  
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    #after 10000 round we get total reward   
    total_reward = total_reward + reward
            
    #total reward by random algorithm - 1200 approx
    #total reward by UCB - 2200 approx (almost double)
    #best ad = index 4 means ad 5.
    
#Visualising the result
plt.hist(ads_selected)
plt.title('Histogram of Ad Selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
