import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
from collections import Counter
import string
import operator
import random
import pickle
import tensorflow as tf

def nmax(num, T, nwords):
    """
    Takes in an array T (of probabilities) and output the top num words
    associated with the highest probabilities in T.
    """
    top_n = T.argsort()[-num:][::-1]
    for n in top_n:
        nwords.append(data['all_words'][n])
    return nwords

os.chdir('txt_sentoken')
data = pickle.load(open("all_data.pkl", "rb"))

## Part 2
# 
p_pos = np.zeros(np.array(data['x_train'][0]).shape)
p_neg = np.zeros(np.array(data['x_train'][0]).shape)
M = []
K = []
T = []
N = []
P = []

#neg, pos, neg, pos, neg...
for n in range(int((array(data['x_train']).shape[0])/2)):  #half of the total

    #negative review
    p_neg += np.array(data['x_train'][2*n])
   
    #positive review
    p_pos += np.array(data['x_train'][2*n+1])
    
for m1 in range(1,26):
    for k1 in range(1,51):
        # for k1 in range(1,21):
    # m = 2
    # k = 1
        k = (0.2)*k1
        m = (0.2)*m1
        Pr_pos = np.zeros(np.array(data['x_train'][0]).shape)
        Pr_neg = np.zeros(np.array(data['x_train'][0]).shape)
        
        for i in range(len(p_neg)):
            Pr_neg[i] = np.log((p_neg[i]+m*k)/(800+k))
            Pr_pos[i] = np.log((p_pos[i]+m*k)/(800+k))
        
        tot = 0
        num_right = 0
        num_neg = 0
        num_pos = 0
        
        for review in range(200):
            
            if data['y_valid'][review] == [0,1]:
                actual = 'NEG'
            else:
                actual = 'POS'
                
            NEG = sum(np.multiply(Pr_neg,np.array(data['x_valid'][review])))
            POS = sum(np.multiply(Pr_pos,np.array(data['x_valid'][review])))
            if NEG > POS:
                guess = 'NEG'
            else:
                guess = 'POS'
            if actual == guess:
                num_right += 1
                if guess == 'NEG':
                    num_neg += 1
                else:
                    num_pos += 1
            tot += 1
            #print(review, actual, guess)
        print('m = ',m,'; k = ',k)
        print("Percent right TOT: ", num_right/tot)
        print("Percent right NEG: ", num_neg/(tot/2))
        print("Percent right POS: ", num_pos/(tot/2))
        K.append(k)
        M.append(m)
        T.append(num_right/tot)
        N.append(num_neg/(tot/2))
        P.append(num_pos/(tot/2))

choices = np.where(np.array(T)==max(T))[0]
highest = 0
highnum = 100000000000000000
for x in choices:
    m = M[x]
    k = K[x]
    
    Pr_pos = np.zeros(np.array(data['x_train'][0]).shape)
    Pr_neg = np.zeros(np.array(data['x_train'][0]).shape)
    
    for i in range(len(p_neg)):
        Pr_neg[i] = np.log((p_neg[i]+m*k)/(800+k))
        Pr_pos[i] = np.log((p_pos[i]+m*k)/(800+k))
    
    tot = 0
    num_right = 0
    num_neg = 0
    num_pos = 0
    
    for review in range(200):
        
        if data['y_test'][review] == [0,1]:
            actual = 'NEG'
        else:
            actual = 'POS'
            
        NEG = sum(np.multiply(Pr_neg,np.array(data['x_test'][review])))
        POS = sum(np.multiply(Pr_pos,np.array(data['x_test'][review])))
        if NEG > POS:
            guess = 'NEG'
        else:
            guess = 'POS'
        if actual == guess:
            num_right += 1
            if guess == 'NEG':
                num_neg += 1
            else:
                num_pos += 1
        tot += 1
        #print(review, actual, guess)
    print('m = ',m,'; k = ',k)
    print("Percent right TOT: ", num_right/tot)
    print("Percent right NEG: ", num_neg/(tot/2))
    print("Percent right POS: ", num_pos/(tot/2))
    if num_right/tot > highest:
        highest = num_right/tot
        highnum = x
    
# m = M[highnum]
# k = K[highnum]
m = 0.1
k = 0.38
Pr_pos = np.zeros(np.array(data['x_train'][0]).shape)
Pr_neg = np.zeros(np.array(data['x_train'][0]).shape)

for i in range(len(p_neg)):
    Pr_neg[i] = np.log((p_neg[i]+m*k)/(800+k))
    Pr_pos[i] = np.log((p_pos[i]+m*k)/(800+k))

tot = 0
num_right = 0
num_neg = 0
num_pos = 0
print('Training Set')
for review in range(1600):
    
    if data['y_train'][review] == [0,1]:
        actual = 'NEG'
    else:
        actual = 'POS'
        
    NEG = sum(np.multiply(Pr_neg,np.array(data['x_train'][review])))
    POS = sum(np.multiply(Pr_pos,np.array(data['x_train'][review])))
    if NEG > POS:
        guess = 'NEG'
    else:
        guess = 'POS'
    if actual == guess:
        num_right += 1
        if guess == 'NEG':
            num_neg += 1
        else:
            num_pos += 1
    tot += 1
    #print(review, actual, guess)
print('m = ',m,'; k = ',k)
print("Percent right TOT: ", num_right/tot)
print("Percent right NEG: ", num_neg/(tot/2))
print("Percent right POS: ", num_pos/(tot/2))

tot = 0
num_right = 0
num_neg = 0
num_pos = 0
print('\nValidation Set')
for review in range(200):
    
    if data['y_valid'][review] == [0,1]:
        actual = 'NEG'
    else:
        actual = 'POS'
        
    NEG = sum(np.multiply(Pr_neg,np.array(data['x_valid'][review])))
    POS = sum(np.multiply(Pr_pos,np.array(data['x_valid'][review])))
    if NEG > POS:
        guess = 'NEG'
    else:
        guess = 'POS'
    if actual == guess:
        num_right += 1
        if guess == 'NEG':
            num_neg += 1
        else:
            num_pos += 1
    tot += 1
    #print(review, actual, guess)
print('m = ',m,'; k = ',k)
print("Percent right TOT: ", num_right/tot)
print("Percent right NEG: ", num_neg/(tot/2))
print("Percent right POS: ", num_pos/(tot/2))

tot = 0
num_right = 0
num_neg = 0
num_pos = 0
print('\nTest Set')
for review in range(200):
    
    if data['y_test'][review] == [0,1]:
        actual = 'NEG'
    else:
        actual = 'POS'
        
    NEG = sum(np.multiply(Pr_neg,np.array(data['x_test'][review])))
    POS = sum(np.multiply(Pr_pos,np.array(data['x_test'][review])))
    if NEG > POS:
        guess = 'NEG'
    else:
        guess = 'POS'
    if actual == guess:
        num_right += 1
        if guess == 'NEG':
            num_neg += 1
        else:
            num_pos += 1
    tot += 1
    #print(review, actual, guess)
print('m = ',m,'; k = ',k)
print("Percent right TOT: ", num_right/tot)
print("Percent right NEG: ", num_neg/(tot/2))
print("Percent right POS: ", num_pos/(tot/2))

## Part 3

print('\nTop 10 words that strongly predict the review is POSITIVE:')
print(nmax(10,p_pos-p_neg,[]))
print('\nTop 10 words that strongly predict the review is NEGATIVE:')
print(nmax(10,p_neg-p_pos,[]))

prob = {}
prob['ppos'] = p_pos
prob['pneg'] = p_neg
pickle.dump(prob, open('prob.pkl', 'wb'))

#return to upper directory
os.chdir('..')
