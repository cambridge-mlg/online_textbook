import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import misc
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'Georgia'

np.random.seed(0)
cov = np.array([[1, 0.995],
                [0.995, 1]])

mu = np.array([0, 0])
norm_rands = np.random.multivariate_normal(mu, cov, 100)

plt.figure(figsize = (12, 4))

plt.subplot(131)
rands = np.random.rand(28, 28)
plt.imshow(rands, origin = 'lower', cmap = 'binary')
plt.title('Random noise', fontsize = 18)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)

plt.subplot(132)
ksi = plt.imread('dimred_ksi.png')
plt.imshow(ksi, cmap = 'binary')
plt.title(r'Greek letter $\xi$', fontsize = 18)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)

plt.subplot(133)
doodle = plt.imread('dimred_picasso.jpg')
plt.imshow(doodle, origin = 'upper', cmap = 'binary')
plt.title(r"Picasso's three musicians", fontsize = 18)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig('dim_red_examples.svg')
plt.show()


plt.subplot(111)
plt.scatter(norm_rands[:, 0], norm_rands[:, 1], marker = 'x', color = 'red')
plt.arrow(0, 0, 0.75, 0.75, color = 'black', width = 0.05)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
plt.title('Highly correlated $2$D data', fontsize = 18)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.gca().set_aspect(1)
plt.tight_layout()
plt.savefig('dim_red_gaussian.svg')
plt.show()
