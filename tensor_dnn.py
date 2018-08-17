import tensorflow as tf
import numpy as np
from main import train_X, train_Y, test_X, test_Y



def initialize_parameters(layers, n_x):
	parameters = {}
	prev_layer = n_x
	for i in range(len(layers)):
		wi = "W" + str(i)
		bi = "b" + str(i)
		parameters[wi] = tf.get_variable(wi, [layers[i], prev_layer], initializer=tf.contrib.layers.xavier_initializer(seed=1))
		parameters[bi] = tf.get_variable(bi, [layers[i], 1], initializer=tf.zeros_initializer())
		prev_layer = layers[i]
	return parameters


def forward_prop(X, parameters, keep_prob=0.8):
	Ai = X
	last_layer = (len(parameters) // 2) - 1
	for i in range(last_layer):
		Wi = parameters["W" + str(i)]
		bi = parameters["b" + str(i)]
		Zi = tf.add(tf.matmul(Wi, Ai), bi)
		#Ai = tf.nn.relu(Zi)
		Ai = tf.nn.dropout(tf.nn.relu(Zi), keep_prob)

	WL = parameters["W" + str(last_layer)]
	bL = parameters["b" + str(last_layer)]
	return tf.add(tf.matmul(WL, Ai), bL) # ZL 

def compute_cost(ZL, Y):
	logits = tf.transpose(ZL)
	labels = tf.transpose(Y)

	cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
	return cost
	

def model(train_X, train_Y, test_X, test_Y, keep_prob = 0.8,learning_rate=0.01, num_epochs=1500, minibatch_size=32, flip=False):
	if flip:
		train_X = train_X.T
		train_Y = train_Y.T
		test_X = test_X.T
		test_Y = test_Y.T

	tf.set_random_seed(1)
	n_x, m = train_X.shape
	print("Input layer has {} units. There are {} training examples".format(n_x, m))
	output_layer = 1
	layers = [100, 100, 10, output_layer]
	
	X = tf.placeholder(tf.float32, shape=[n_x, None])
	Y = tf.placeholder(tf.float32, shape=[output_layer, None])

	parameters = initialize_parameters(layers, n_x)

	ZL = forward_prop(X, parameters)
	cost = compute_cost(ZL, Y)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


	init = tf.global_variables_initializer() # look into this


	with tf.Session() as sess:
		sess.run(init)
		
		for i in range(num_epochs):
			feed_dict= {X:train_X, Y:train_Y}
			_, curr_cost = sess.run([optimizer, cost], feed_dict)
			if i % 1000 == 0: 
				print("Cost at step {} is {}".format(i, curr_cost))

		# Save parameters, predict test set 
		parameters = sess.run(parameters)
		print("Parameters saved!")

		preds = tf.round(tf.sigmoid(ZL))
		set_preds = tf.equal(preds, Y)
		accuracy = tf.reduce_mean(tf.cast(set_preds, "float"))
		print("Training accuracy {}".format(accuracy.eval({X: train_X, Y: train_Y})))
		print("Test accuracy {}".format(accuracy.eval({X: test_X, Y: test_Y})))


		sess.close()

model(train_X, train_Y, test_X, test_Y, flip=True, num_epochs=10001, learning_rate=0.01)

