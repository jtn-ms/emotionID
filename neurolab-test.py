import numpy as np
import neurolab as nl

target = [[-1, -1,  1, -1, -1,\
	   -1,  1,  1, -1, -1,\
	   -1, -1,  1, -1, -1,\
	   -1, -1,  1, -1, -1,\
	   -1, -1,  1, -1, -1,\
	   -1, -1,  1, -1, -1,\
	   -1,  1,  1,  1, -1],\
	  [-1,  1,  1,  1, -1,\
	    1, -1, -1, -1,  1,\
	   -1, -1, -1, -1,  1,\
	   -1, -1, -1,  1, -1,\
	   -1, -1,  1, -1, -1,\
	   -1,  1, -1, -1, -1,\
	    1,  1,  1,  1,  1]]

input2 = [[-1,  1,  1, -1, -1,\
	    1, -1, -1, -1,  1,\
	   -1, -1, -1, -1,  1,\
	   -1, -1, -1,  1, -1,\
	   -1, -1,  1, -1, -1,\
	   -1,  1, -1, -1, -1,\
	    1,  1,  1,  1,  1],\
	  [-1,  1,  1, -1, -1,\
	    1, -1, -1, -1,  1,\
	   -1, -1, -1, -1,  1,\
	   -1, -1, -1,  1, -1,\
	   -1, -1, -1, -1, -1,\
	   -1,  1, -1, -1, -1,\
	    1,  1,  1,  1,  1]]
#hop field
hop = nl.net.newhop(target)
output = hop.sim(input2)
#print(output)
#hemming
hem = nl.net.newhop(target)
output = hem.sim(input2)
#print(np.argmax(output, axis=0))
#print(output)
#feedforward
# Create train samples
x = np.array(target)
y = np.array([0,1])
insize =  x.shape[1]
size = x.shape[0]
inp = x.reshape(size,35)
tar = y.reshape(size,1)

# Create network with 2 layers and random initialized
inrange = np.ones((2 * insize,))
inrange = inrange.reshape(insize,2)
for i in range(insize):
    inrange[i][0] = -1
net = nl.net.newff(inrange,[20, 1])

# Train network
print x
print y
error = net.train(inp, tar, epochs=5000, show=100, goal=0.001)

# Simulate network
out = net.sim(inp)

# Plot result
import pylab as pl
pl.subplot(211)
pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('error (default SSE)')
pl.show()

x2 = np.array(input2)
y2 = net.sim(x2.reshape(x2.shape[0],35)).reshape(x2.shape[0])

print x2
print y2

'''
https://pythonhosted.org/neurolab/lib.html#module-neurolab.train
https://pythonhosted.org/neurolab/ex_newff.html
'''
