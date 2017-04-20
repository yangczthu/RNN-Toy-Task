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
		elif line[0] == "T":
			continue
		else:
			flag = flag + 1
			# if flag % 5 == 0:
			line = line.strip()
			data = line.split('\t')
			steps.append(float(data[0]))
			losses.append(float(data[1]))
	
	return steps, losses, name

def move_avg(a,n=10):
	ret = np.cumsum(a)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n-1:]/n


if __name__ == '__main__':
	arg = sys.argv


	# T = int(arg[1])
	legend = []
	l = len(arg) - 1
	mylist=range(1, l+1)
	for k in range(l):
		i = arg[k + 1]
		steps, losses, name = read_file(i)
		
		tempval = move_avg(losses)
		mylist[k] = np.array(tempval)
		#print(len(tempval))
		#print(len(steps))
		plt.plot(steps[9:], tempval)
		legend.append(name)


	plt.title('Multiplication digit Task')
	plt.ylabel('MSE')
	plt.xlabel('Training iterations')
	plt.axis([0,100000,0,0.3])
	plt.legend(legend, loc='upper right')
	plt.show()
	# from matplotlib.backends.backend_pdf import PdfPages
	# pp = PdfPages('mnist.pdf')
	# plt.savefig(pp, format='pdf')
	# pp.close()