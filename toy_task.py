'''
Don't believe everything below... 
'''


from __future__ import absolute_import
from __future__ import division
from __future__	import print_function

import numpy as np
import argparse, os
import tensorflow as tf

from tensorflow.contrib.rnn import BasicLSTMCell
from toy_data import prepare_data

def next_batch(train_data):
	return None


def main(model, n_iter, n_batch, n_hidden):


	# --- Set data params ----------------

	n_input = 10 * 4
	n_output = 4 * 4

	n_classes = 5 

	# --- Prepare data -------------
	train_data, test_data = prepare_data()


	# --- Create graph and compute gradients ----------------------
	x = tf.placeholder("float", [None, n_input, n_classes])
	y = tf.placeholder("float", [None, n_output, n_classes])
	


	V_init_val = np.sqrt(6.)/np.sqrt(n_classes * 2)



	# --- Input to hidden layer ----------------------
	cell = BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)
	hidden_out, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
	# --- Hidden Layer to Output ----------------------

	V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_classes], \
			dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
	V_bias = tf.get_variable("V_bias", shape=[n_classes], \
			dtype=tf.float32, initializer=tf.constant_initializer(0.01))

	hidden_out_list = tf.unstack(hidden_out, axis=1)
	temp_out = tf.stack([tf.matmul(i, V_weights) for i in hidden_out_list[-n_output:]])
	output_data = tf.nn.bias_add(tf.transpose(temp_out, [1,0,2]), V_bias) 

	# --- evaluate process ----------------------
	mse = tf.reduce_mean(tf.squared_difference(y, output_data))


	# --- Initialization ----------------------
	optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9).minimize(mse)
	init = tf.global_variables_initializer()


	# --- Training Loop ----------------------


	with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)) as sess:

		sess.run(init)

		step = 0
		steps = []
		mses = []

		while step < n_iter:
			batch_x, batch_y = next_batch(n_batch)

			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

			mse_value = sess.run(mse, feed_dict={x: batch_x, y: batch_y})

			print("Iter " + str(step) + ", MSE= " + "{:.6f}".format(mse_value))

			steps.append(step)
			mses.append(mse_value)

			step += 1
		print("Optimization Finished!")

if __name__=="__main__":
	parser = argparse.ArgumentParser(
		description="RNN Toy Task")
	parser.add_argument("model", default='LSTM', help='Model name: LSTM, EURNN, uLSTM, resNet')
	parser.add_argument('--n_iter', '-I', type=int, default=10000, help='training iteration number')
	parser.add_argument('--n_batch', '-B', type=int, default=128, help='batch size')
	parser.add_argument('--n_hidden', '-H', type=int, default=1024, help='hidden layer size')

	args = parser.parse_args()
	dict = vars(args)

	for i in dict:
		if (dict[i]=="False"):
			dict[i] = False
		elif dict[i]=="True":
			dict[i] = True
		
	kwargs = {	
				'model': dict['model'],
				'n_iter': dict['n_iter'],
			  	'n_batch': dict['n_batch'],
			  	'n_hidden': dict['n_hidden'],
			}

	main(**kwargs)
