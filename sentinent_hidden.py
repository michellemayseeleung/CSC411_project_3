import os
import numpy as np
from pylab import *
from collections import Counter
import matplotlib.pyplot as plt
import string
import operator
import random
import pickle
import tensorflow as tf

## Internal functions

def get_train_batch(x_train, y_train, size):
    x_batch = []
    y_batch = []
    indices_one = random.sample(range(1, 600, 2), size//2)
    indices_zero = random.sample(range(0, 600, 2), size//2)
    for i in indices_one:
        x_batch.append(x_train[i])
        y_batch.append(y_train[i])
    for i in indices_zero:
        x_batch.append(x_train[i])
        y_batch.append(y_train[i])
    
    return x_batch, y_batch

## Part 1
os.chdir("txt_sentoken")

# global dictionary of data
M = {}
neg_train = []
neg_valid = []
neg_test = []
pos_train = []
pos_valid = []
pos_test = []

# negative reviews
os.chdir("neg")
# tracks which dataset to insert into
i = 0
# tracks all words appearing in all negative reviews
neg_master = Counter()
for fn in os.listdir("."):
    c = Counter()
    for line in open(fn):
        words = line.split()
        
        # remove single-character punctuation (doesn't affect it's, i'm, etc)
        words = [x for x in words if x not in string.punctuation]
        c.update(words)
        neg_master.update(words)
    i += 1
    if i >= 601:
        if i >= 801:
            neg_test.append(c)
        else:
            neg_valid.append(c)
    else:
        neg_train.append(c)

M["neg_train"] = neg_train
M["neg_valid"] = neg_valid
M["neg_test"] = neg_test

# sorts the frequency dictionary into a list
sorted_neg = sorted(neg_master.items(), key=operator.itemgetter(1), reverse=True)
os.chdir("..")

# positive reviews
os.chdir("pos")
i = 0
pos_master = Counter()
for fn in os.listdir("."):
    c = Counter()
    for line in open(fn):
        words = line.split()
        words = [x for x in words if x not in string.punctuation]
        c.update(words)
        pos_master.update(words)
        
    i += 1
    if i >= 601:
        if i >= 801:
            pos_test.append(c)
        else:
            pos_valid.append(c)
    else:
        pos_train.append(c)

M["pos_train"] = pos_train
M["pos_valid"] = pos_valid
M["pos_test"] = pos_test

sorted_pos = sorted(pos_master.items(), key=operator.itemgetter(1), reverse=True)
os.chdir("..")

# number of keywords to note
top_kek = 200
neg = sorted_neg[:top_kek]    
pos = sorted_pos[:top_kek]    
neg_keywords = array(neg).T[0]
pos_keywords = array(pos).T[0]

print("Negative keywords: ")
for i in range(top_kek):
    if (neg[i][0] not in pos_keywords):
        print(neg[i])

print("Positive keywords: ")
for i in range(top_kek):
    if (pos[i][0] not in neg_keywords):
        print(pos[i])

## Part 4

# get k, the size of all words appearing in all reviews
master = pos_master.copy()
master.update(neg_master)

# create a list of all the words, list is ordered so we can iterate through it and 
# mark words as appear or not appear in each review
all_words = list(master.keys())
k = len(master)
assert k == len(all_words)

# pre-process data and pickle it for future use since this step takes a long time
# and we don't want to have to reprocess each time it's run
if not os.path.exists("all_data.pkl"):
    data = {}
    x_train = []
    x_valid = []
    x_test = []
    y_train = []
    y_valid = []
    y_test = []
    
    for i in range(600):
        review = list(neg_train[i].keys())
        x = [1 if all_words[j] in review else 0 for j in range(k)]
        x_train.append(x)
        review = list(pos_train[i].keys())
        x = [1 if all_words[j] in review else 0 for j in range(k)]
        x_train.append(x)
        
        y_train.append([0, 1])    
        y_train.append([1, 0])
    
    
    for i in range(200):
        review = list(neg_valid[i].keys())
        x = [1 if all_words[j] in review else 0 for j in range(k)]
        x_valid.append(x)
        review = list(pos_valid[i].keys())
        x = [1 if all_words[j] in review else 0 for j in range(k)]
        x_valid.append(x)
        
        y_valid.append([0, 1])    
        y_valid.append([1, 0])
        
    
    for i in range(200):
        review = list(neg_test[i].keys())
        x = [1 if all_words[j] in review else 0 for j in range(k)]
        x_test.append(x)
        review = list(pos_test[i].keys())
        x = [1 if all_words[j] in review else 0 for j in range(k)]
        x_test.append(x)
        
        y_test.append([0, 1])    
        y_test.append([1, 0])
        
    data["x_train"] = x_train
    data["x_valid"] = x_valid
    data["x_test"] = x_test
    data["y_train"] = y_train
    data["y_valid"] = y_valid
    data["y_test"] = y_test
                
    pickle.dump(data, open("all_data.pkl", "wb"))
else:
    data = pickle.load(open("all_data.pkl", "rb"))
    x_train = data["x_train"]
    x_valid = data["x_valid"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_valid= data["y_valid"]
    y_test = data["y_test"]

# for loop to determine best regularization factor lambda
lambdas = []
for j in range(5):
    print ("Lambda = ", 0.00003 * j + 0.00001)
    # set up inputs 
    x_in = tf.placeholder(tf.float32, [None, k], name="x_input")
    y_ = tf.placeholder(tf.float32, [None, 2], name="y_input")
    
    seed = 1234
    # initialize weights and bias using truncated normal
    W0 = tf.Variable(tf.truncated_normal([k, 2], stddev=0.01, seed=seed))
    b0 = tf.Variable(tf.truncated_normal([2], stddev=0.01, seed=seed))
    
    layer1 = tf.matmul(x_in, W0)+b0

    y = tf.nn.softmax(layer2)
    
    # weight penalty L2
    lam = 0.0000001 * (10**j)
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))
    reg_NLL = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y))+decay_penalty)
    
    train_step = tf.train.AdamOptimizer(0.0004).minimize(reg_NLL)
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    for i in range(200):
        x_batch, y_batch = get_train_batch(x_train, y_train, 50)
        sess.run(train_step, feed_dict={x_in: x_batch, y_: y_batch})
    
        if i % 25 == 0:
            print("iter: ", i)
            valid_accuracy = sess.run(accuracy, feed_dict={x_in: x_valid, y_: y_valid})
            train_accuracy = sess.run(accuracy, feed_dict={x_in: x_train, y_: y_train})
    
            print("Validation:", valid_accuracy)
            print("Train:", train_accuracy)
            print("Penalty:", sess.run(decay_penalty))
            
    lambdas.append([j, valid_accuracy])

best_lambda = lambdas[np.argmax(lambdas, axis = 0)[1]][0]
    
'''train using validation lambda'''

# set up inputs 
x_in = tf.placeholder(tf.float32, [None, k], name="x_input")
y_ = tf.placeholder(tf.float32, [None, 2], name="y_input")

seed = 1234
# initialize weights and bias using truncated normal
W0 = tf.Variable(tf.truncated_normal([k, nhid], stddev=0.01, seed=seed))
b0 = tf.Variable(tf.truncated_normal([nhid], stddev=0.01, seed=seed))

layer1 = tf.nn.relu(tf.matmul(x_in, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1

y = tf.nn.softmax(layer2)

# weight penalty L2
lam = 0.00003 * best_lambda + 0.00001
decay_penalty =lam*tf.reduce_sum(tf.square(W0)) +lam*tf.reduce_sum(tf.square(W1))
reg_NLL = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y))+decay_penalty)

train_step = tf.train.AdamOptimizer(0.0004).minimize(reg_NLL)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

learning_curve = []

for i in range(200):
    x_batch, y_batch = get_train_batch(x_train, y_train, 50)
    sess.run(train_step, feed_dict={x_in: x_batch, y_: y_batch})

    if i % 25 == 0:
        print("iter: ", i)
        valid_accuracy = sess.run(accuracy, feed_dict={x_in: x_valid, y_: y_valid})
        train_accuracy = sess.run(accuracy, feed_dict={x_in: x_train, y_: y_train})
        test_accuracy = sess.run(accuracy, feed_dict={x_in: x_test, y_: y_test})

        print("Train:", train_accuracy)
        print("Validation:", valid_accuracy)
        print("Test:", test_accuracy)
        print("Penalty:", sess.run(decay_penalty))
        learning_curve.append([i, train_accuracy, valid_accuracy, test_accuracy])

learning_curve = array(learning_curve).T

# plot and save learning curves
plt.figure()
plt.plot(learning_curve[0], learning_curve[1] * 100, 'b-', label = 'training set performance')
plt.plot(learning_curve[0], learning_curve[2] * 100, 'g-', label = 'validation set performance')
plt.plot(learning_curve[0], learning_curve[3] * 100, 'r-', label = 'test set performance')
plt.title("Learning curve with lambda = %s" %(0.00003 * best_lambda + 0.00001))
plt.xlabel("Number of iterations")
plt.ylabel("Percentage of correct classifications")
plt.legend(loc = 'center')
plt.savefig("part4.png")

os.chdir("..")