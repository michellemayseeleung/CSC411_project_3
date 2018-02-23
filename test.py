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
w = pickle.load(open("wb.pkl", "rb"))['w']
b = pickle.load(open("wb.pkl", "rb"))['b']

w_pos = np.zeros(len(w))
w_neg = np.zeros(len(w))

for i in range(len(w)):
    w_pos[i] = w[i][0]-w[i][1]
    w_neg[i] = w[i][1]-w[i][0]
    

print('\nTop 10 words that strongly predict the review is POSITIVE:')
print(nmax(100, w_pos, []))
print('\nTop 10 words that strongly predict the review is NEGATIVE:')
print(nmax(100, w_neg, []))

#return to upper directory
os.chdir('..')