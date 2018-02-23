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
    if (i < 500):
        y = vstack((y, array([0])))
    else:
        y = vstack((y, array([1])))
    
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
    
x_in = tf.placeholder(tf.float32, [None, k], name="x_input")
y_ = tf.placeholder(tf.float32, [None, 1], name="y_input")

# initialize weights and bias using truncated normal
W0 = tf.Variable(tf.truncated_normal([k, 1], stddev=0.01, seed=t))
b0 = tf.Variable(tf.truncated_normal([1], stddev=0.01, seed=t))
    
l = tf.matmul(x_in, W0)+b0

# weight penalty L2
lam = 0.00003
decay_penalty =lam*tf.reduce_sum(tf.square(W0))
reg_NLL = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(l) + (1-y_) * tf.log(1-l))+decay_penalty)

train_step = tf.train.AdamOptimizer(0.0004).minimize(reg_NLL)

correct_prediction = tf.equal(tf.round(l), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

learning_curve = []

for i in range(1501):
    sess.run(train_step, feed_dict={x_in: x, y_: y})

    if i % 50 == 0:
        print("iter: ", i)
        train_accuracy = sess.run(accuracy, feed_dict={x_in: x, y_: y})
        test_accuracy = sess.run(accuracy, feed_dict={x_in: xt, y_: yt})

        print("Train:", train_accuracy)
        print("Test:", test_accuracy)
        print("Penalty:", sess.run(decay_penalty))
        learning_curve.append([i, train_accuracy, test_accuracy])

learning_curve = array(learning_curve).T

# plot and save learning curves
plt.figure()
plt.plot(learning_curve[0], learning_curve[1] * 100, 'b-', label = 'training set performance')
plt.plot(learning_curve[0], learning_curve[2] * 100, 'r-', label = 'test set performance')
plt.title("Learning curve with lambda = %s" %(lam))
plt.xlabel("Number of iterations")
plt.ylabel("Percentage of correct classifications")
plt.legend(loc = 'center')
plt.savefig("part9.png")

## NB

neg = array([x[i] for i in range(1000) if y[i] == 0])
pos = array([x[i] for i in range(1000) if y[i] == 1])