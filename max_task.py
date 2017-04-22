from __future__ import absolute_import
from __future__ import division
from __future__	import print_function

import numpy as np
import argparse, os
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell
from EURNN import EURNNCell


def digit_to_pixel(n):
	if n == 0:
		return [[1]*5, [1] + [0]*3 + [1], [1]*5, [0]*5]
	elif n == 1:
		return [[0]*5, [1]*5, [0]*5, [0]*5]
	elif n == 2:
		return [[1,0,1,1,1], [1,0,1,0,1], [1,1,1,0,1], [0]*5]
	elif n == 3:
		return [[1,0,1,0,1], [1,0,1,0,1], [1,1,1,1,1], [0]*5]
	elif n == 4:
		return [[1,1,1,0,0], [0,0,1,0,0], [1,1,1,1,1], [0]*5]
	elif n == 5:
		return [[1,1,1,0,1], [1,0,1,0,1], [1,0,1,1,1], [0]*5]
	elif n == 6:
		return [[1,1,1,1,1], [1,0,1,0,1], [1,0,1,1,1], [0]*5]
	elif n == 7:
		return [[1,0,0,0,0], [1,0,0,0,0], [1,1,1,1,1], [0]*5]
	elif n == 8:
		return [[1,1,1,1,1], [1,0,1,0,1], [1,1,1,1,1], [0]*5]
	elif n == 9:
		return [[1,1,1,0,1], [1,0,1,0,1], [1,1,1,1,1], [0]*5]
	# +
	elif n == 10:
		return [[0,0,1,0,0], [0,1,1,1,0], [0,0,1,0,0], [0]*5]
	# =
	elif n == 11:
		return [[0,1,0,1,0], [0,1,0,1,0], [0,1,0,1,0], [0]*5]
	# blank
	elif n == 12:
		return [[0]*5, [0]*5, [0]*5, [0]*5]
	# *
	elif n == 13:
		return [[0,1,0,1,0], [0,0,1,0,0], [0,1,0,1,0], [0]*5]

def max_data(n, math, ratio):

	x = []
	y = []

	if math == 'ADD':
		for i in range(n):
			d1 = random.randint(0, 4)
			d2 = random.randint(0, 9)
			d3 = random.randint(0, 9)
			d4 = random.randint(0, 4)
			d5 = random.randint(0, 9)
			d6 = random.randint(0, 9)
			x_element = []
			x_element += digit_to_pixel(d1)
			x_element += digit_to_pixel(d2)
			x_element += digit_to_pixel(d3)
			x_element += digit_to_pixel(10)
			x_element += digit_to_pixel(d4)
			x_element += digit_to_pixel(d5)
			x_element += digit_to_pixel(d6)
			x_element += digit_to_pixel(11)
			x_element += digit_to_pixel(12)
			x_element += digit_to_pixel(12)
			x_element += digit_to_pixel(12)


			tmp = d1 * 100 + d4 * 100 + d2 * 10 + d5 * 10 + d3 + d6
			y1 = int(tmp/100)
			y2 = int((tmp%100)/10)
			y3 = tmp%10

			y_element = []
			y_element += digit_to_pixel(y1)
			y_element += digit_to_pixel(y2)
			y_element += digit_to_pixel(y3)


			x.append(x_element)
			y.append(y_element)
	elif math == 'MULTIPLY':
		for i in range(n):
			d1 = random.randint(0, 9)
			d2 = random.randint(0, 9)
			d3 = random.randint(0, 9)
			d4 = random.randint(0, 9)
			x_element = []
			x_element += digit_to_pixel(d1)
			x_element += digit_to_pixel(d2)
			x_element += digit_to_pixel(13)
			x_element += digit_to_pixel(d3)
			x_element += digit_to_pixel(d4)
			x_element += digit_to_pixel(11)
			x_element += digit_to_pixel(12)
			x_element += digit_to_pixel(12)
			x_element += digit_to_pixel(12)
			x_element += digit_to_pixel(12)


			tmp = (d1 * 10 + d2) * (d3 * 10 + d4)
			y1 = int(tmp/1000)
			y2 = int((tmp%1000)/100)
			y3 = int((tmp%100)/10)
			y4 = int(tmp%10)

			y_element = []
			y_element += digit_to_pixel(y1)
			y_element += digit_to_pixel(y2)
			y_element += digit_to_pixel(y3)
			y_element += digit_to_pixel(y4)

			x.append(x_element)
			y.append(y_element)

	elif math == 'MIX':
		for i in range(n):
			d1 = random.randint(0, 9)
			d2 = random.randint(0, 9)
			d3 = random.randint(0, 9)
			d4 = random.randint(0, 9)
			d5 = random.randint(0, 9)
			d6 = random.randint(0, 9)
			x_element = []
			x_element += digit_to_pixel(d1)
			x_element += digit_to_pixel(d2)
			x_element += digit_to_pixel(d3)
			if random.random() > ratio:
				s = False
				x_element += digit_to_pixel(10)
			else:
				s = True
				x_element += digit_to_pixel(13)
			x_element += digit_to_pixel(d4)
			x_element += digit_to_pixel(d5)
			x_element += digit_to_pixel(d6)
			x_element += digit_to_pixel(11)
			x_element += digit_to_pixel(12)
			x_element += digit_to_pixel(12)
			x_element += digit_to_pixel(12)
			x_element += digit_to_pixel(12)
			x_element += digit_to_pixel(12)			
			x_element += digit_to_pixel(12)

			n1 = d1 * 100 + d2 * 10 + d3
			n2 = d4 * 100 + d5 * 10 + d6

			if s:
				o = n1 * n2
			else:
				o = n1 + n2


			y1 = int(o/100000)
			y2 = int((o%100000)/10000)
			y3 = int((o%10000)/1000)
			y4 = int((o%1000)/100)
			y5 = int((o%100)/10)
			y6 = int(o%10)

			y_element = []
			y_element += digit_to_pixel(y1)
			y_element += digit_to_pixel(y2)
			y_element += digit_to_pixel(y3)
			y_element += digit_to_pixel(y4)
			y_element += digit_to_pixel(y5)			
			y_element += digit_to_pixel(y6)

			x.append(x_element)
			y.append(y_element)

	x = np.array(x).astype(np.float32)
	y = np.array(y).astype(np.float32)

	return x, y



def main(model, math, n_iter, n_batch, n_hidden, capacity, comp, FFT):
	print(math)
	n_test = 10000
	# --- Set data params ----------------
	if math == 'ADD':
		n_input = 11 * 4
		n_output = 3 * 4
	elif math == 'MULTIPLY':
		n_input = 10 * 4
		n_output = 4 * 4
	elif math == 'MIX':
		n_input = 14 * 4
		n_output = 6 * 4

	n_classes = 5

  	# --- Create data --------------------


	# --- Create graph and compute gradients ----------------------
	x = tf.placeholder("float", [None, n_input, n_classes])
	y = tf.placeholder("float", [None, n_output, n_classes])
	

	V_init_val = np.sqrt(6.)/np.sqrt(n_classes * 2)



	# --- Input to hidden layer ----------------------
	if model == "LSTM":
		cell = BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)
		hidden_out, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
	elif model == "RNN":
		cell = BasicRNNCell(n_hidden)
		hidden_out, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
	elif model == "EURNN":
		cell = EURNNCell(n_hidden, capacity, FFT, comp)
		if comp:
			hidden_out_comp, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.complex64)
			hidden_out = tf.real(hidden_out_comp)
		else:
			hidden_out, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
	# --- Hidden Layer to Output ----------------------

	V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_classes], \
			dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
	V_bias = tf.get_variable("V_bias", shape=[n_classes], \
			dtype=tf.float32, initializer=tf.constant_initializer(0.01))

	hidden_out_list = tf.unstack(hidden_out, axis=1)
	temp_out = tf.stack([tf.matmul(i, V_weights) for i in hidden_out_list[-n_output:]])
	print(temp_out)
	output_data = tf.nn.bias_add(tf.transpose(temp_out, [1,0,2]), V_bias) 

	# --- evaluate process ----------------------
	mse = tf.reduce_mean(tf.squared_difference(y, output_data))


	# --- Initialization ----------------------
	optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9).minimize(mse)
	init = tf.global_variables_initializer()



	# --- save result ----------------------
	filename = "./output/max/"  + model + "_N=" + str(n_hidden)
		
	if math == 'ADD':
		filename += '_add'
	elif math == 'MULTIPLY':
		filename += '_multiply'
	elif math == 'MIX':
		filename += '_mix'
		
	if model == "EURNN" or model == "uLSTM":
		if FFT:
			filename += "_FFT"
		else:
			filename = filename + "_L=" + str(capacity)

	filename = filename + ".txt"
	if not os.path.exists(os.path.dirname(filename)):
		try:
			os.makedirs(os.path.dirname(filename))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	f = open(filename, 'w')
	f.write("########\n\n")
	f.write("## \tModel: %s with N=%d"%(model, n_hidden))
	if model == "EURNN" or model == "uLSTM":
		if FFT:
			f.write(" FFT")
		else:
			f.write(" L=%d"%(capacity))
	f.write("\n\n")
	f.write("########\n\n")

	# --- Training Loop ----------------------


	with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)) as sess:

		sess.run(init)

		step = 0
		steps = []
		mses = []

		while step < n_iter:
			train_ratio = 0.1
			batch_x, batch_y = max_data(n_batch, math, train_ratio)

			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

			mse_value = sess.run(mse, feed_dict={x: batch_x, y: batch_y})

			print("Iter " + str(step) + ", MSE= " + "{:.6f}".format(mse_value))

			steps.append(step)
			mses.append(mse_value)
			# accs.append(acc)				

			# if step % 1000 == 500:
			# 	x_data = sess.run(x, feed_dict={x: batch_x, y: batch_y})[0]
			# 	y_data = sess.run(output_data, feed_dict={x: batch_x, y: batch_y})[0]
			# 	fig = plt.figure()
			# 	fig.add_subplot(1,2,1)
			# 	plt.imshow(x_data.transpose())
			# 	fig.add_subplot(1,2,2)
			# 	plt.imshow(y_data.transpose())
			# 	plt.show()


			f.write("%d\t%f\n"%(step, mse_value))

			step += 1
		print("Optimization Finished!")


		x_data = sess.run(x, feed_dict={x: batch_x, y: batch_y})[0]
		y_data = sess.run(output_data, feed_dict={x: batch_x, y: batch_y})[0]
		fig = plt.figure()
		fig.add_subplot(1,2,1)
		plt.imshow(x_data.transpose())
		fig.add_subplot(1,2,2)
		plt.imshow(y_data.transpose())
		from matplotlib.backends.backend_pdf import PdfPages
		if math == "ADD":
			pp = PdfPages('adding.pdf')
		elif math == "MULTIPLY":
			pp = PdfPages('multiplication.pdf')
		elif math == "MIX":
			pp = PdfPages('mix.pdf')

		plt.savefig(pp, format='pdf')
		pp.close()


		
		# --- test ----------------------
		test_ratio = -1
		test_x, test_y = max_data(n_test, math, test_ratio)
		test_mse = sess.run(mse, feed_dict={x: test_x, y: test_y})
		print("Adding test result: MSE= " + "{:.6f}".format(test_mse))

		test_ratio = 10
		test_x, test_y = max_data(n_test, math, test_ratio)
		test_mse = sess.run(mse, feed_dict={x: test_x, y: test_y})
		print("Multiplying test result: MSE= " + "{:.6f}".format(test_mse))

if __name__=="__main__":
	parser = argparse.ArgumentParser(
		description="Max Task")
	parser.add_argument("model", default='LSTM', help='Model name: LSTM, EURNN, uLSTM, resNet')
	parser.add_argument("math", default='ADD', help='ADD or MULTIPLY or MIX')
	parser.add_argument('--n_iter', '-I', type=int, default=100000, help='training iteration number')
	parser.add_argument('--n_batch', '-B', type=int, default=128, help='batch size')
	parser.add_argument('--n_hidden', '-H', type=int, default=1024, help='hidden layer size')
	parser.add_argument('--capacity', '-L', type=int, default=2, help='Tunable style capacity, only for EURNN, default value is 2')
	parser.add_argument('--comp', '-C', type=str, default="False", help='Complex domain or Real domain. Default is False: real domain')
	parser.add_argument('--FFT', '-F', type=str, default="False", help='FFT style, only for EURNN, default is False')

	args = parser.parse_args()
	dict = vars(args)

	for i in dict:
		if (dict[i]=="False"):
			dict[i] = False
		elif dict[i]=="True":
			dict[i] = True
		
	kwargs = {	
				'model': dict['model'],
				'math': dict['math'],
				'n_iter': dict['n_iter'],
			  	'n_batch': dict['n_batch'],
			  	'n_hidden': dict['n_hidden'],
			  	'capacity': dict['capacity'],
			  	'comp': dict['comp'],
			  	'FFT': dict['FFT'],
			}

	main(**kwargs)
