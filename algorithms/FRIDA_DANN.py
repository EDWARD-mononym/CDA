import argparse
import os
import numpy as np
import math
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

root_dir = root_dir = "gdrive/My Drive/data_set_da/OfficeHome_ResFeatures/officehome_resnet50/"

so_=np.loadtxt(os.path.join(root_dir, "RealWorld_RealWorld.csv"), delimiter=',') 

S_feature = so_[:,:2048]   #np.loadtxt(os.path.join(root_dir, "RealWorld_RealWorld.csv"), delimiter=',')  
S_feature_l = so_[:,2048]  #np.loadtxt(os.path.join(root_dir, "Product_labels_train_1sep.csv"), delimiter=',') 

feat = S_feature
lab = S_feature_l

lab = np.array(lab, np.int)
lab = np.expand_dims(lab, 1)
feat, lab = shuffle(feat, lab)

lab_oh = np.eye(65)[lab]

lab_oh = np.reshape(lab_oh, [lab_oh.shape[0], lab_oh.shape[1]*lab_oh.shape[2]])
print(lab_oh.shape)

feat.shape

mb_size = 64
Z_dim = 5000
X_dim = feat.shape[1]
y_dim = lab_oh.shape[1]
h_dim = 512
num_epoch= 1000

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.sigmoid(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob

""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, feat.shape[1]])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

D_W3 = tf.Variable(xavier_init([h_dim, y_dim]))
D_b3 = tf.Variable(tf.zeros(shape=[y_dim]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    
    D_class_logit = tf.matmul(D_h1, D_W3) + D_b3
    D_class_prob = tf.nn.softmax(D_class_logit)

    return D_prob, D_logit, D_class_prob, D_class_logit

def next_batch(s, e, input1, l1):
    
    inp1 = input1[s:e,:]
    
    lab1 = l1[s:e,:]

        
    return inp1, lab1

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])