import os
import numpy as np
from pylab import *
from collections import Counter
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

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations

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
        
        y_train.append([0])    
        y_train.append([1])
    
    
    for i in range(200):
        review = list(neg_valid[i].keys())
        x = [1 if all_words[j] in review else 0 for j in range(k)]
        x_valid.append(x)
        review = list(pos_valid[i].keys())
        x = [1 if all_words[j] in review else 0 for j in range(k)]
        x_valid.append(x)
        
        y_valid.append([0])    
        y_valid.append([1])
        
    
    for i in range(200):
        review = list(neg_test[i].keys())
        x = [1 if all_words[j] in review else 0 for j in range(k)]
        x_test.append(x)
        review = list(pos_test[i].keys())
        x = [1 if all_words[j] in review else 0 for j in range(k)]
        x_test.append(x)
        
        y_test.append([0])    
        y_test.append([1])
        
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

# set up summary and logging    
with tf.name_scope('input'):
    x_in = tf.placeholder(tf.float32, [None, k], name="x_input")
    y_ = tf.placeholder(tf.float32, [None, 1], name="y_input")


seed = 1234
nhid = 1
# initialize weights and bias using truncated normal

with tf.name_scope("weights"):
    W0 = tf.Variable(tf.truncated_normal([k, nhid], stddev=0.01, seed=seed))
    variable_summaries(W0)
with tf.name_scope("bias"):
    b0 = tf.Variable(tf.truncated_normal([nhid], stddev=0.01, seed=seed))
    variable_summaries(b0)

W1 = tf.Variable(tf.truncated_normal([nhid, 1], stddev=0.01, seed=seed))
b1 = tf.Variable(tf.truncated_normal([1], stddev=0.01, seed=seed))

# layer1 = tf.nn.tanh(tf.matmul(x_in, W0)+b0)
with tf.name_scope("wx+b"):
    layer2 = tf.matmul(x_in, W0)+b0
    tf.summary.histogram("layer2", layer2)

y = tf.relu(layer2)

# weight penalty L2
lam = 0.00000
decay_penalty =lam*tf.reduce_sum(tf.square(W0)) #+lam*tf.reduce_sum(tf.square(W1))
reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

train_step = tf.train.AdamOptimizer(0.0004).minimize(reg_NLL)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.round(y), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

learning_curve = []
writer = tf.summary.FileWriter("logs", graph=tf.get_default_graph())

for i in range(1000):
    x_batch, y_batch = get_train_batch(x_train, y_train, 50)
    sess.run(train_step, feed_dict={x_in: x_batch, y_: y_batch})

    if i % 25 == 0:
        print("i=",i)
        test_accuracy = sess.run(accuracy, feed_dict={x_in: x_test, y_: y_test})
        valid_accuracy = sess.run(accuracy, feed_dict={x_in: x_valid, y_: y_valid})
        train_accuracy = sess.run(accuracy, feed_dict={x_in: x_train, y_: y_train})

        print("Test:", test_accuracy)
        print("Validation:", valid_accuracy)
        print("Train:", train_accuracy)
        print("Penalty:", sess.run(decay_penalty))
        
        learning_curve.append([i, train_accuracy, valid_accuracy, test_accuracy])

learning_curve = array(learning_curve).T

os.chdir("..")