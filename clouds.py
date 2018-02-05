from __future__ import division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random, math

# Load images into np arrays and trim dimensions
# Satelite x, y, rgb ; mask x, y binary
labelImg = Image.open('images/mask1.png')
satImg = Image.open('images/satellite1.png')

mask = np.delete(np.array(labelImg), [1,2,3], 2)
sat = np.delete(np.array(satImg), 3, 2)
# convert 255's to 1's
mask[mask > 1] = 1

# NxN patch size
N = 3
b = int((N - 1) / 2)

# Loop through image matrix and store patches of size NxN in list
data = []
labels = []

dataSize = (sat.shape[0] - b * 2) * (sat.shape[1] - b * 2)
test = np.empty((dataSize, N * N * sat.shape[2]))
print test.shape

i = 0
for x in range(b,sat.shape[0] - b):
	for y in range(b, sat.shape[1] - b):
		i =+ 1
		data.append(np.ndarray.flatten(sat[x-b:x+b+1 ,y-b : y+b+1]))
		labels.append(mask[x,y])

data = np.array(data)
labels = np.array(labels)

# Calculate train and test size and randomize sets
train_size = 0.9
n_smpls = len(data)
n_train = int(math.ceil(n_smpls * train_size))
n_test = n_smpls - n_train

indices = range(n_smpls)




x = tf.placeholder('float', [None, N * N * sat.shape[2]])
y = tf.placeholder('float')


def neural_net_model(data):
	n_nodes_1 = 200
	n_nodes_2 = 200
	n_nodes_3 = 200

	batch_size = 100

	hidden_layer_1 = {'wheigts':tf.Variable(tf.random_normal([data.shape[1], n_nodes_1])),
					  'biases':tf.Variable(tf.random_normal(n_nodes_1))}

	hidden_layer_2 = {'wheigts':tf.Variable(tf.random_normal([n_nodes_1, n_nodes_2])),
					  'biases':tf.Variable(tf.random_normal(n_nodes_2))}

	hidden_layer_3 = {'wheigts':tf.Variable(tf.random_normal([n_nodes_2, n_nodes_3])),
					  'biases':tf.Variable(tf.random_normal(n_nodes_3))}

	output_layer = {'wheigts':tf.Variable(tf.random_normal([n_nodes_3, 1])),
					  'biases':tf.Variable(tf.random_normal(1))}

	l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
	l2 = tf.nn.relu(l2)labels

	l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']), output_layer['biases']

	return output


def train_neural_network(x):
	
	prediction = neural_net_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	n_epochs = 10

	with tf.Sessions() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in n_epochs:
			epoch_loss = 0
			# for _ in range(int())