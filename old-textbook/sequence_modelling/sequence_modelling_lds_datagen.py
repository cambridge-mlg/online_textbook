import numpy as np
import matplotlib.pyplot as plt

sig_noise = 1
sig_trans = 1
lamda = 0.9
coeff = 1

no_points = 100
xs = [np.random.normal(0, sig_trans)]
ys = [np.random.normal(coeff*xs[-1], sig_noise)]

for n in range(no_points):
    x_ = np.random.normal(lamda*xs[-1], sig_trans)
    y_ = np.random.normal(coeff*xs[-1], sig_noise)
    
    xs.append(x_)
    ys.append(y_)


plt.scatter(np.arange(len(ys)), ys, marker='o', linewidth=1.5, edgecolor='black')
plt.show()
