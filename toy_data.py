import numpy as np
import random

random.seed(1234)

def _digit_to_pixel(n):
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


def _all_data():
	x = []
	y = []

	for a in range(100):
		for b in range(100):
			d1 = a/10
			d2 = a%10
			d3 = b/10
			d4 = b%10

			x_element = []
			x_element += _digit_to_pixel(d1)
			x_element += _digit_to_pixel(d2)
			x_element += _digit_to_pixel(13)
			x_element += _digit_to_pixel(d3)
			x_element += _digit_to_pixel(d4)
			x_element += _digit_to_pixel(11)
			x_element += _digit_to_pixel(12)
			x_element += _digit_to_pixel(12)
			x_element += _digit_to_pixel(12)
			x_element += _digit_to_pixel(12)


			out = a * b
			y1 = int(out/1000)
			y2 = int((out%1000)/100)
			y3 = int((out%100)/10)
			y4 = int(out%10)

			y_element = []
			y_element += _digit_to_pixel(y1)
			y_element += _digit_to_pixel(y2)
			y_element += _digit_to_pixel(y3)
			y_element += _digit_to_pixel(y4)

			x.append(x_element)
			y.append(y_element)


	return np.array(x), np.array(y)

def prepare_data():
	x, y = _all_data()
	test_ind = random.sample(range(10000), 1000)
	train_val_ind = list(set(range(10000)) - set(test_ind))
	test_data = x[test_ind], y[test_ind]
	train_val_data = x[train_val_ind], y[train_val_ind]
	return train_val_data, test_data


if __name__ == '__main__':
	train_val_data, test_data = prepare_data()
	x, y = test_data
	print x[1], y[1]


