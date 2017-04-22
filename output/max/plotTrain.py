import sys
import matplotlib.pyplot as plt
import numpy as np

def read_file(filename):
	print filename
	f = open(filename, "r")
	steps = []
	losses = []
	acc = []
	flag = 0
	for line in f:
		if line[0] == "#":
			if line[2] == " ":
				name = line.split('\t')[1]
				name = name[7: -1]
			continue
		elif line[0] == " ":
			continue
		elif line[0] == "\t":
			continue
		elif line[0] == "\n":
			continue
		else:
			flag = flag + 1
			# if flag % 5 == 0:
			line = line.strip()
			data = line.split('\t')
			steps.append(float(data[0]))
			losses.append(float(data[1]))
			# acc.append(float(data[2]))
	
	return steps, losses, name


if __name__ == '__main__':
	arg = sys.argv
	color_list = ['b', 'm', 'r', '#7a6a1a', 'g']

	# T = int(arg[1])
	legend = []
	l = len(arg) - 1
	for k in range(l):
		i = arg[k + 1]
		steps, losses, name = read_file(i)
		plt.plot(steps, losses)
		legend.append(name)
		

	# base = 10*np.log(8)/(200+20)	
	# plt.plot(steps, [base]*len(steps), ':')
	plt.title('Max Task')
	plt.ylabel('MSE')
	plt.xlabel('Training iterations')
	plt.axis([0,10000,0,0.2])

	plt.legend(legend, loc='lower right')
	plt.show()


	# from matplotlib.backends.backend_pdf import PdfPages
	# pp = PdfPages('mnist.pdf')
	# plt.savefig(pp, format='pdf')
	# pp.close()