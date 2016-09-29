# coding: utf-8
'''
This script intends to draw training loss and testing error.You can either find the 
log file under /tmp/caffe.localhost.localdomain1.pengmin.log.INFO.20160708-205951.9906
or redirect it while training 'caffe train -solver solver.prototxt 2>&1 | tee log.txt'.
'''

import numpy as np
import re

import matplotlib.pyplot as plt
import pylab
from pylab import figure, show, legend
from mpl_toolkits.axes_grid1 import host_subplot

fp = open('../log.txt', 'r')
loss_png_name = raw_input("Please input the savefile name: ")

train_iterations = []
train_loss = []
test_iterations = []
test_error = []

for line in fp:
	if '] Iteration' in line and 'loss = ' in line:
		arr = re.findall(r'ion \b\d+\b,', line) # lists like ['ion 100,']
		train_iterations.append(int(arr[0].strip(',')[4:]))
		train_loss.append(float(line.strip().split(' = ')[-1]))
	if '] Iteration' in line and 'Testing net (#0)' in line:
		arr = re.findall(r'ion \b\d+\b,', line)
		test_iterations.append(int(arr[0].strip(',')[4:]))
	if '#' in line and 'error' in line:
		test_error.append(float(line.strip().split(' = ')[-1].split()[0]))
fp.close()

host = host_subplot(111)
plt.subplots_adjust(right=0.8)
par1 = host.twinx()
host.set_xlabel('iterations')
host.set_ylabel('train loss')
par1.set_ylabel('test error')

p1, = host.plot(train_iterations, train_loss, label="training log loss")
p2, = par1.plot(test_iterations, test_error, label="test error")
host.legend(loc=1)

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())

host.set_xlim([10000, 50000])
host.set_ylim([0.001, 0.01])
par1.set_ylim([0.002, 0.006])

plt.draw()
plt.savefig(loss_png_name)