import numpy as np
import GPy
import matplotlib.pyplot as plt

GPy.plotting.change_plotting_library('matplotlib')

handle = open('mauna_data.txt')
x = []
y = []

for line in handle:
    
    split = line.split()
    x.append(float(split[0]))
    y.append(float(split[1]))

x, y = np.array(x), np.array(y)

x = np.reshape(x, (-1, 1))
y = np.reshape(y, (-1, 1))

rbf = GPy.kern.RBF(input_dim=1, variance=1., lengthscale= 1)
bias = GPy.kern.Bias(input_dim = 1, variance=1.0, active_dims=None, name='bias')

m = GPy.models.GPRegression(x, y, rbf*bias)
m.optimize(messages = True)
#plt.scatter(x, y, marker = 'o', color = 'black', s = 1)

m.plot_confidence(color = 'black')
m.plot_mean(color = 'black')
m.plot_data(color = 'black', marker = 'o', s = 1)

m.plot()
plt.show()
