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
np.random.seed(1)
# Load the word data
data = pickle.load(open("all_data.pkl", "rb"))

# Load the thetas (probabilities) for Naive Bayes
p_pos = pickle.load(open("prob.pkl", "rb"))['ppos']
p_neg = pickle.load(open("prob.pkl", "rb"))['pneg']

# Load the thetas (weights and biases) for Logistic Regression
w = pickle.load(open("wb.pkl", "rb"))['w']
b = pickle.load(open("wb.pkl", "rb"))['b']

#generate index list of positive words
top_pos = (p_pos-p_neg).argsort()[-20:][::-1]

#generate index list of negative words
top_neg = (p_neg-p_pos).argsort()[-20:][::-1]

x = []
y = []
for i in range(800):
    sub = zeros(len(data['x_train'][0]))
    sub[i] = 1
    vstack((x,sub))
    if i < 400:
        y.append([1,0])
    else:
        y.append([0,1])

#return to upper directory
os.chdir('..')