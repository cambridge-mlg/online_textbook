import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *
import matplotlib.image as image
import os
import math
set_notebook_preferences()

def free_energy(x, K, q, mu, pi, cov):
    
    prec = np.linalg.inv(cov)
    exponent = np.einsum('ijk, nik -> nij', prec, x - mu)
    exponent = -1/2*np.einsum('nij, nij -> ni', x - mu, exponent)

    log_joint = exponent + np.log(pi) - 0.5*np.log(2*np.pi*np.linalg.det(cov))

    f_energy = np.sum(q*log_joint - q*np.log(q))
    
    return f_energy
    

def mog_EM(x, K, q_init, mu_init, pi_init, cov_init, max_steps):
    
    xs = np.stack([x]*K, axis = 1)
    q, mu, pi, cov = q_init.copy(), mu_init.copy(), pi_init.copy(), cov_init.copy()
    f_energy = free_energy(xs, K, q_init, mu_init, pi_init, cov_init)
    qs, mus, pis, covs, f_energies = [q.T], [mu], [pi], [cov], [f_energy]
    
    for n in range(max_steps + 1):
        
        # E-step
        prec = np.linalg.inv(cov)
        exponent = np.einsum('ijk, nik -> nij', prec, xs - mu)
        exponent = np.einsum('nij, nij -> ni', xs - mu, exponent)
        exp_term = np.exp(-1/2*exponent)
        q = (exp_term*pi/(2*np.pi*np.linalg.det(cov))**0.5).T
        q = q/q.sum(axis = 0)
        
        qs.append(q), mus.append(mu), pis.append(pi), covs.append(cov)
        f_energies.append(free_energy(xs, K, q.T, mu, pi, cov))

        # M-step
        N_k = np.sum(q, axis = 1)
        mu = (q.dot(x).T/N_k).T

        pi = N_k/x.shape[0]

        cov = np.einsum('ijk, ijl -> ijkl', xs - mu, xs - mu)
        cov = np.einsum('ij, jikl -> ijkl', q, cov)
        cov = (np.sum(cov, axis = 1).T/N_k).T
        
        qs.append(q), mus.append(mu), pis.append(pi), covs.append(cov)
        f_energies.append(free_energy(xs, K, q.T, mu, pi, cov))
    
    return qs, mus, pis, covs, f_energies

x = np.load('clustering_2d.npy') # load the 2d clustering dataset
im = image.imread('skull.png')
K = 3 
q_init = np.ones(shape = (x.shape[0], K))/K # initial responsibilities = 1/K
mu_init = np.array([[1.1, -1.7], [-0.7, -1], [1, 0.75]]) # initial means judged by eye
cov_init = np.stack([np.eye(2)/10, np.eye(2)/10, np.eye(2)*2]) # initial covariances isotropic but with different widths
pi_init = np.ones(shape = (K,))/K # initial cluster probabilities = 1/K
max_steps = 10 # maximum number of EM steps

qs, mus, pis, covs, f_energies = mog_EM(x, K, q_init, mu_init, pi_init, cov_init, max_steps) # apply EM
last_energy = np.array(f_energies)[np.where(np.logical_not(np.isnan(f_energies)))][-1]
last_idx = np.where(np.logical_not(np.isnan(f_energies)))[0][-1]

x_, y_ = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 250)) # x, y where to evaluate gaussians
grid = np.stack([x_, y_], axis = -1) # x, y grid to use for calculation of gaussians
colors = ['red', 'green', 'blue']

os.chdir('clust_collapse')
for i in range(2*max_steps + 1):
    
    plt.close()
    plt.figure(figsize = (10, 5))
    
    plt.subplot(1, 2, 1)
    q, mu, pi, cov = qs[i], mus[i], pis[i], covs[i] # parameters of the i^th EM step
    
    for k in range(K):
        mu_, cov_ = mu[k], cov[k] # mean/covariance of the k^th gaussian cluster

        exp_ = np.exp(-0.5*np.sum((grid - mu_).dot(np.linalg.inv(cov_))*(grid - mu_),
                                  axis = -1)) # evaluate the k^th gaussian on the grid
        plt.contour(x_, y_, exp_, 4, colors = colors[k]) # plot the gaussian

    plt.scatter(x[:, 0], x[:, 1], s = 20, c = q.T, edgecolor = 'black') # show data, coloured by membership q
    beautify_plot({"title":"Iter. {} ({} step)".format((i+1)//2, 'EM'[(i + 1)%2]), "x":"$x$", "y":"$y$"})

    plt.subplot(1, 2, 2)
    for j in range(2*max_steps):
        colour = ['purple', 'orange'][j%2]
        if j == 0:
            plt.plot(np.arange(j, j+2), f_energies[j:j+2], color = colour, label = 'E steps')
        elif j == 1:
            plt.plot(np.arange(j, j+2), f_energies[j:j+2], color = colour, label = 'M steps')
        else:
            plt.plot(np.arange(j, j+2), f_energies[j:j+2], color = colour)
            plt.plot(np.arange(j, j+2), f_energies[j:j+2], color = colour)

    plt.plot(np.arange(i, 2*max_steps + 1), f_energies[i:2*max_steps + 1],
             color = 'white', linewidth = 4)
        
    plt.gca().legend(loc = 4, fontsize = 15)
    plt.scatter(0, f_energies[0], marker = 'x', color = 'black', zorder = 4)
    plt.ylim([-1650, 0])
    plt.xticks(np.arange(0, 7, 2))
    if i > last_idx:
        extent = [last_idx - 0.15, last_idx + 0.15, last_energy - 50, last_energy + 50]
        plt.gca().imshow(im, aspect='auto', extent = extent, zorder=11)
    beautify_plot({"title":"Optimisation of $\mathcal{F}$", "x":"Number of E and M steps", "y":"Free energy $\mathcal{F}$"})
    plt.xlim([-0.5, 6])
    plt.tight_layout()
    plt.savefig('{}.png'.format(str(i).zfill(3)), dpi = 300)
