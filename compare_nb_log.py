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
    values = []
    top_n = T.argsort()[-num:][::-1]
    for n in top_n:
        nwords.append(((data['all_words'][n])))
        values.append(round(T[n],3))
    return nwords
    
os.chdir('txt_sentoken')

# Load the word data
data = pickle.load(open("all_data.pkl", "rb"))

# Load the thetas (probabilities) for Naive Bayes
p_pos = pickle.load(open("prob.pkl", "rb"))['ppos']
p_neg = pickle.load(open("prob.pkl", "rb"))['pneg']

# Load the thetas (weights and biases) for Logistic Regression
w = pickle.load(open("wb.pkl", "rb"))['w']
b = pickle.load(open("wb.pkl", "rb"))['b']

w_pos = np.zeros(len(w))
w_neg = np.zeros(len(w))

for i in range(len(w)):
    w_pos[i] = (w[i][0]+b[0])-(w[i][1]+b[1])
    w_neg[i] = (w[i][1]+b[1])-(w[i][0]+b[0])

log_pos = nmax(100, w_pos, [])
# log_neg = nmax(100, w_neg, [])
nb_pos = nmax(100,(p_pos/800)-(p_neg/800),[])
# nb_neg = nmax(100,p_neg-p_pos,[])

print('\nTop 100 words LOGISTIC REGRESSION:')
print(log_pos)
# print('\nTop 100 words that strongly predict the review is NEGATIVE:')
# print(log_neg)
# 
print('\nTop 100 words NAIVE BAYES:')
print(nb_pos)
# print('\nTop 100 words that strongly predict the review is NEGATIVE:')
# print(nb_neg)


both = [list(set(log_pos).intersection(nb_pos))]#,list(set(log_neg).intersection(nb_neg))]
log_only = [list(set(log_pos)-set(nb_pos))]#,list(set(log_neg)-set(nb_neg))]
nb_only = [list(set(nb_pos)-set(log_pos))]#,list(set(nb_neg)-set(log_neg))]

print('POSITIVE words that appear in both')
print(both[0])
print('\nPOSITIVE words that appear only in logistic regression')
print(log_only[0])
print('\nPOSITIVE words that appear only in naive bayes')
print(nb_only[0])
# 
# print('\nNEGATIVE words that appear in both')
# print(both[1])
# print('\nNEGATIVE words that appear only in logistic regression')
# print(log_only[1])
# print('\nNEGATIVE words that appear only in naive bayes')
# print(nb_only[1])
# 
# 
# w_max =np.zeros(len(w))
# for i in range(len(w)):
#     w_max[i] = max((w[i][0]+b[0]),(w[i][1]+b[1]))
#     
# p_max = np.zeros(len(p_pos))
# for i in range(len(p_pos)):
#     p_max[i] = max(p_pos[i]/800,p_neg[i]/800)
# 
# logmax = nmax(100,w_max,[])
# nbmax = nmax(100,p_max,[])
# 
# bothmax = [list(set(logmax).intersection(nbmax))]
# logmax_only = [list(set(logmax)-set(nbmax))]
# nbmax_only = [list(set(nbmax)-set(logmax))]
# 
# print('Words that appear in both')
# print(both[1])
# print('\nWords that appear only in logistic regression')
# print(log_only[1])
# print('\nWords that appear only in naive bayes')
# print(nb_only[1])

#return to upper directory
os.chdir('..')