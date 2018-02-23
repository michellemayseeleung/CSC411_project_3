import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import collections
import string
import operator
import random
import pickle
import tensorflow as tf
import time
import scipy.spatial.distance as ssd

## Part 7
os.chdir('txt_sentoken')

data = load('embeddings.npz')
emb = data['emb']
indices = data['word2ind'].flatten()[0]

word2ind = dict([[v,k] for k,v in indices.items()])
vocab = word2ind.keys()

# for every w in emb(?), if output pair t is close to w, result = 1
# word2vec -> map each word into k dim vector ([left context, right context], w)
# emb -> list of words with their respective embeddings

x_train = {}
for word in vocab:
    x_train[word] = set()
x_valid = {}
for word in vocab:
    x_valid[word] = set()
x_test = {}
for word in vocab:
    x_test[word] = set()

## negatives
os.chdir('neg')
r = 0
for fn in os.listdir("."):
    r += 1
    # 100 reviews each to train
    if (r > 100):
        break
    for line in open(fn):
        words = line.split()
        for i in range(len(words)):
            if (i == 0):
                if (len(words) > 1):
                    if (words[0] in vocab and words[1] in vocab):
                        x_train[words[0]].add(words[1])
            elif (i == len(words) - 1):
                if (len(words) > 1):
                    if (words[i] in vocab and words[i-1] in vocab):
                        x_train[words[i]].add(words[i-1])
            else:
                if (words[i] in vocab):
                    if (words [i-1] in vocab):
                        x_train[words[i]].add(words[i-1])
                    if (words [i+1] in vocab):
                        x_train[words[i]].add(words[i+1])

for fn in os.listdir("."):
    r += 1
    if (r <= 100):
        continue
    if (r > 110):
        break
    for line in open(fn):
        words = line.split()
        for i in range(len(words)):
            if (i == 0):
                if (len(words) > 1):
                    if (words[0] in vocab and words[1] in vocab):
                        x_test[words[0]].add(words[1])
            elif (i == len(words) - 1):
                if (len(words) > 1):
                    if (words[i] in vocab and words[i-1] in vocab):
                        x_test[words[i]].add(words[i-1])
            else:
                if (words[i] in vocab):
                    if (words [i-1] in vocab):
                        x_test[words[i]].add(words[i-1])
                    if (words [i+1] in vocab):
                        x_test[words[i]].add(words[i+1])

for fn in os.listdir("."):
    r += 1
    if (r <= 110):
        continue
    if (r > 120):
        break
    for line in open(fn):
        words = line.split()
        for i in range(len(words)):
            if (i == 0):
                if (len(words) > 1):
                    if (words[0] in vocab and words[1] in vocab):
                        x_valid[words[0]].add(words[1])
            elif (i == len(words) - 1):
                if (len(words) > 1):
                    if (words[i] in vocab and words[i-1] in vocab):
                        x_valid[words[i]].add(words[i-1])
            else:
                if (words[i] in vocab):
                    if (words [i-1] in vocab):
                        x_valid[words[i]].add(words[i-1])
                    if (words [i+1] in vocab):
                        x_valid[words[i]].add(words[i+1])
os.chdir('..')

## positives
os.chdir('pos')
r = 0
for fn in os.listdir("."):
    r += 1
    # 100 reviews each to train
    if (r > 100):
        break
    for line in open(fn):
        words = line.split()
        for i in range(len(words)):
            if (i == 0):
                if (len(words) > 1):
                    if (words[0] in vocab and words[1] in vocab):
                        x_train[words[0]].add(words[1])
            elif (i == len(words) - 1):
                if (len(words) > 1):
                    if (words[i] in vocab and words[i-1] in vocab):
                        x_train[words[i]].add(words[i-1])
            else:
                if (words[i] in vocab):
                    if (words [i-1] in vocab):
                        x_train[words[i]].add(words[i-1])
                    if (words [i+1] in vocab):
                        x_train[words[i]].add(words[i+1])
                        
for fn in os.listdir("."):
    r += 1
    if (r <= 100):
        continue
    if (r > 110):
        break
    for line in open(fn):
        words = line.split()
        for i in range(len(words)):
            if (i == 0):
                if (len(words) > 1):
                    if (words[0] in vocab and words[1] in vocab):
                        x_test[words[0]].add(words[1])
            elif (i == len(words) - 1):
                if (len(words) > 1):
                    if (words[i] in vocab and words[i-1] in vocab):
                        x_test[words[i]].add(words[i-1])
            else:
                if (words[i] in vocab):
                    if (words [i-1] in vocab):
                        x_test[words[i]].add(words[i-1])
                    if (words [i+1] in vocab):
                        x_test[words[i]].add(words[i+1])
                        
for fn in os.listdir("."):
    r += 1
    if (r <= 110):
        continue
    if (r > 120):
        break
    for line in open(fn):
        words = line.split()
        for i in range(len(words)):
            if (i == 0):
                if (len(words) > 1):
                    if (words[0] in vocab and words[1] in vocab):
                        x_valid[words[0]].add(words[1])
            elif (i == len(words) - 1):
                if (len(words) > 1):
                    if (words[i] in vocab and words[i-1] in vocab):
                        x_valid[words[i]].add(words[i-1])
            else:
                if (words[i] in vocab):
                    if (words [i-1] in vocab):
                        x_valid[words[i]].add(words[i-1])
                    if (words [i+1] in vocab):
                        x_valid[words[i]].add(words[i+1])
os.chdir('..')

## format training data
x = []
y = []
list_vocab = list(vocab)

t = 12974189123
random.seed(t)

p = time.time()

for word in x_train:
    context = x_train[word]
    num_not = len(context)
    not_nearby = [item for item in list_vocab if item not in context]
    for nearby in context:
        x.append([word, nearby])
        y.append([1])
    for i in range(num_not):
        x.append([word, random.choice(not_nearby)])
        y.append([0])

print(time.time() - p)

for i in range(len(x)):
    x[i] = [emb[word2ind[x[i][0]]], emb[word2ind[x[i][1]]]]

_x = np.reshape(array(x), (array(x).shape[0], 256))
_y = array(y)

## validation set
xv = []
yv = []
p = time.time()
for word in x_valid:
    context = x_valid[word]
    num_not = len(context)
    not_nearby = [item for item in list_vocab if item not in context]
    for nearby in context:
        xv.append([word, nearby])
        yv.append([1])
    for i in range(num_not):
        xv.append([word, random.choice(not_nearby)])
        yv.append([0])

for i in range(len(xv)):
    xv[i] = [emb[word2ind[xv[i][0]]], emb[word2ind[xv[i][1]]]]

_xv = np.reshape(array(xv), (array(xv).shape[0], 256))
_yv = array(yv)

print(time.time() - p)

## test set
xt = []
yt = []
p = time.time()
for word in x_test:
    context = x_test[word]
    num_not = len(context)
    not_nearby = [item for item in list_vocab if item not in context]
    for nearby in context:
        xt.append([word, nearby])
        yt.append([1])
    for i in range(num_not):
        xt.append([word, random.choice(not_nearby)])
        yt.append([0])

for i in range(len(xt)):
    xt[i] = [emb[word2ind[xt[i][0]]], emb[word2ind[xt[i][1]]]]

_xt = np.reshape(array(xt), (array(xt).shape[0], 256))
_yt = array(yt)

print(time.time() - p)

## Logistic regression
def get_train_batch(_x, _y, size):
    idx = array(np.random.permutation(_x.shape[0])[:size])
    x_batch = _x[idx]
    y_batch = _y[idx]
    return x_batch, y_batch


for z in range(10):
    
    x_in = tf.placeholder(tf.float32, [None, 256], name="x_input")
    y_ = tf.placeholder(tf.float32, [None, 1], name="y_input")
    
    # initialize weights and bias using truncated normal
    W0 = tf.Variable(tf.truncated_normal([256, 1], stddev=0.01, seed=t))
    b0 = tf.Variable(0.1)
        
    l = tf.matmul(x_in, W0)+b0
    
    # weight penalty L2
    lam = 0.00003 * z
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))
    reg_NLL = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(l) + (1-y_) * tf.log(1-l))+decay_penalty)
    
    train_step = tf.train.AdamOptimizer(0.0004).minimize(reg_NLL)
    
    correct_prediction = tf.equal(tf.round(l), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    learning_curve = []
    
    for i in range(1001):
        xb, yb = get_train_batch(_x, _y, 25000)
        sess.run(train_step, feed_dict={x_in: xb, y_: yb})
    
        if i % 25 == 0:
            print("iter: ", i)
            train_accuracy = sess.run(accuracy, feed_dict={x_in: _x, y_: _y})
            valid_accuracy = sess.run(accuracy, feed_dict={x_in: _xv, y_: _yv})
            test_accuracy = sess.run(accuracy, feed_dict={x_in: _xt, y_: _yt})
    
            print("Train:", train_accuracy)
            print("Train:", valid_accuracy)
            print("Test:", test_accuracy)
            print("Penalty:", sess.run(decay_penalty))
    
            learning_curve.append([i, train_accuracy, valid_accuracy, test_accuracy])
    
    learning_curve = array(learning_curve).T

    # plot and save learning curves
    plt.figure()
    plt.plot(learning_curve[0], learning_curve[1] * 100, 'b-', label = 'training set performance')
    plt.plot(learning_curve[0], learning_curve[2] * 100, 'g-', label = 'validation set performance')
    plt.plot(learning_curve[0], learning_curve[3] * 100, 'r-', label = 'test set performance')
    plt.title("Learning curve with lambda = %s" %(lam))
    plt.xlabel("Number of iterations")
    plt.ylabel("Percentage of correct classifications")
    plt.legend(loc = 'center')
    plt.savefig("part7-" + str(z) + ".png")

os.chdir('..')

## Part 8

def similarity(word):
    cos_sim = []
    index = word2ind[word]
    embedding = emb[index]
    
    for i in range(emb.shape[0]):
        if (i != index):
            cos_sim.append(ssd.cosine(embedding, emb[i]))
        else:
            cos_sim.append(float('inf'))
            
    cos_like = np.argsort(array(cos_sim))[:10]     
    
    print("CLOSE TO" + word)
    for index in cos_like:
        print(indices[index])    
    print('\n\r')

similarity('good')
similarity('story')
similarity('movie')
similarity('boring')