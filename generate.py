import os
import numpy as np
from collections import Counter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import string
import operator
import random
import pickle
import tensorflow as tf

k = 100

## training set
x = zeros((0, 100))
y = zeros((0, 1))

t = 2274192891
np.random.seed(t)

for i in range(970):
    row = np.random.randint(2, size=98)
    row = np.append(row, array([0,0]))
    row = row.T
    x = vstack((x, row))
    y = vstack((y, np.random.randint(2, size=1)))
    
for i in range(15):
    row = np.random.randint(2, size=98)
    row = np.append(row, array([1,0]))
    row = row.T
    x = vstack((x, row))
    y = vstack((y, array([1])))
    
for i in range(15):
    row = np.random.randint(2, size=98)
    row = np.append(row, array([0,1]))
    row = row.T
    x = vstack((x, row))
    y = vstack((y, array([1])))
    
## test set
xt = zeros((0, 100))
yt = zeros((0, 1))

for i in range(30):
    row = np.random.randint(2, size=98)
    row = np.append(row, array([0,0]))
    row = row.T
    xt = vstack((xt, row))
    yt = vstack((yt, array([0])))

for i in range(15):
    row = np.random.randint(2, size=98)
    row = np.append(row, array([1,0]))
    row = row.T
    xt = vstack((xt, row))
    yt = vstack((yt, array([1])))
    
for i in range(15):
    row = np.random.randint(2, size=98)
    row = np.append(row, array([0,1]))
    row = row.T
    xt = vstack((xt, row))
    yt = vstack((yt, array([1])))
#     
# x_in = tf.placeholder(tf.float32, [None, k], name="x_input")
# y_ = tf.placeholder(tf.float32, [None, 1], name="y_input")
# 
# # initialize weights and bias using truncated normal
# W0 = tf.Variable(tf.truncated_normal([k, 1], stddev=0.01, seed=t))
# b0 = tf.Variable(tf.truncated_normal([1], stddev=0.01, seed=t))
#     
# l = tf.matmul(x_in, W0)+b0
# 
# # weight penalty L2
# lam = 0.00003
# decay_penalty =lam*tf.reduce_sum(tf.square(W0))
# reg_NLL = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(l) + (1-y_) * tf.log(1-l))+decay_penalty)
# 
# train_step = tf.train.AdamOptimizer(0.0004).minimize(reg_NLL)
# 
# correct_prediction = tf.equal(tf.round(l), y_)
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# 
# learning_curve = []
# 
# for i in range(1501):
#     sess.run(train_step, feed_dict={x_in: x, y_: y})
# 
#     if i % 50 == 0:
#         print("iter: ", i)
#         train_accuracy = sess.run(accuracy, feed_dict={x_in: x, y_: y})
#         test_accuracy = sess.run(accuracy, feed_dict={x_in: xt, y_: yt})
# 
#         print("Train:", train_accuracy)
#         print("Test:", test_accuracy)
#         print("Penalty:", sess.run(decay_penalty))
#         learning_curve.append([i, train_accuracy, test_accuracy])
# 
# learning_curve = array(learning_curve).T
# 
# # plot and save learning curves
# plt.figure()
# plt.plot(learning_curve[0], learning_curve[1] * 100, 'b-', label = 'training set performance')
# plt.plot(learning_curve[0], learning_curve[2] * 100, 'r-', label = 'test set performance')
# plt.title("Learning curve with lambda = %s" %(lam))
# plt.xlabel("Number of iterations")
# plt.ylabel("Percentage of correct classifications")
# plt.legend(loc = 'center')
# plt.savefig("part9.png")

## NB
M = []
K = []
T = []
N = []
P = []
p_neg = zeros(array(x).shape[1])
p_pos = zeros(array(x).shape[1])

for n in range(int((array(x).shape[0]))):  #half of the total
    if y[n] == 0: #neg
        p_neg += np.array(x[n])
    else: #pos
        p_pos += np.array(x[n])
    
for m1 in range(1,2):
    for k1 in range(1,5):
        # for k1 in range(1,21):
    # m = 2
    # k = 1
        k = (0.2)*k1
        m = (0.2)*m1
        Pr_pos = np.zeros(np.array(x[0]).shape)
        Pr_neg = np.zeros(np.array(x[0]).shape)
        
        for i in range(len(p_neg)):
            Pr_neg[i] = np.log((p_neg[i]+m*k)/(1000+k))
            Pr_pos[i] = np.log((p_pos[i]+m*k)/(1000+k))
        
        tot = 0
        num_right = 0
        num_neg = 0
        num_pos = 0
        
        for review in range(60):
            
            if yt[review] == 0:
                actual = 'NEG'
            else:
                actual = 'POS'
                
            NEG = sum(np.multiply(Pr_neg,np.array(xt[review])))
            POS = sum(np.multiply(Pr_pos,np.array(xt[review])))
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
            print(review, actual, guess)
        print('m = ',m,'; k = ',k)
        print("Percent right TOT: ", num_right/tot)
        print("Percent right NEG: ", num_neg/(tot/2))
        print("Percent right POS: ", num_pos/(tot/2))
        K.append(k)
        M.append(m)
        T.append(num_right/tot)
        N.append(num_neg/(tot/2))
        P.append(num_pos/(tot/2))

