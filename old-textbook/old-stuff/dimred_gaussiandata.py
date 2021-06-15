import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'Georgia'

cov = np.array([[1, 0.995],
                [0.995, 1]])

mu = np.array([0, 0])


rands = np.random.multivariate_normal(mu, cov, 100)

plt.figure(figsize = (5, 5))
plt.scatter(rands[:, 0], rands[:, 1], marker = 'x', color = 'red')
plt.arrow(0, 0, 0.75, 0.75, color = 'black', width = 0.05)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
plt.title('Highly correlated $2$D data', fontsize = 18)
plt.show()
