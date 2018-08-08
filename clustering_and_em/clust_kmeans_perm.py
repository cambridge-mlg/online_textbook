import numpy as np
from helper_functions import *
set_notebook_preferences()

np.random.seed(0)
fig = plt.figure(figsize = (12, 4))
axes = []

x = np.load('clustering_2d.npy')

K, N = 3, x.shape[0]

mus = np.array([[-0.5, -1.5],
                [-1.5, -1.5],
                [-0.2, 0.8]])
s = np.zeros(shape = (N, K)) # set all membership indices to 0
memberships = np.random.choice(np.arange(0, K), N) # randomly choose a cluster for each point
s[np.arange(s.shape[0]), memberships] = 1 # set the 1-entries according to the random sample
colors = ['r', 'g', 'b']

plt.subplot(1, 3, 1)

for idx, mu in enumerate(mus): # cycle over clusters
    plt.scatter(mu[0], mu[1], marker = '^', color = colors[idx], s = 200,
                edgecolor = 'black', zorder = 2, linewidth = '2') # plot cluster center
    
    points_in_class = x[np.where(s[:, idx] == 1)[0], :] # select points in current cluster ...
    
    plt.scatter(points_in_class[:, 0], points_in_class[:, 1], marker = 'o',
                color = colors[idx], edgecolor = 'black') # ... and plot
axes.append(plt.gca())
beautify_plot({"title":"Initialisation", "x":"$x_1$", "y":"$x_2$"})
plt.xticks(np.arange(-2, 3)), plt.yticks(np.arange(-2, 3))
plt.ylim([-2.5, 2.5])

s, mu_, losses = k_means(x, K, 20, mus)

plt.subplot(1, 3, 2)
for idx, mu in enumerate(mu_):
    plt.scatter(mu[0], mu[1], marker = '^', color = colors[idx], s = 200,
                edgecolor = 'black', zorder = 2, linewidth = '2')
    
    points_in_class = x[np.where(s[:, idx] == 1)[0], :]
    
    plt.scatter(points_in_class[:, 0], points_in_class[:, 1], marker = 'o',
                color = colors[idx], edgecolor = 'black', zorder = 1)

axes.append(plt.gca())
beautify_plot({"title":"After convergence", "x":"$x_1$", "y":"$x_2$"})
plt.xticks(np.arange(-2, 3)), plt.yticks(np.arange(-2, 3))
plt.ylim([-2.5, 2.5])

plt.subplot(1, 3, 3)
plt.plot(np.arange(len(losses)), losses, color = 'black')
beautify_plot({"title":"Optimisation of $\mathcal{C}$", "x":"No. iterations", "y":"Cost $\mathcal{C}$"})

ax0tr = axes[0].transData # Axis 0 -> Display
ax1tr = axes[1].transData # Axis 1 -> Display
figtr = fig.transFigure.inverted() # Display -> Figure
# 2. Transform arrow start point from axis 0 to figure coordinates
ptB = figtr.transform(ax0tr.transform((1, -0.2)))
# 3. Transform arrow end point from axis 1 to figure coordinates
ptE = figtr.transform(ax1tr.transform((-1.8, -0.2)))
# 4. Create the patch
arrow = matplotlib.patches.FancyArrowPatch(
    ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
    fc = "orange", connectionstyle="arc3,rad=0.2", arrowstyle='simple', alpha = 1,
    mutation_scale = 40.)
# 5. Add patch to list of objects to draw onto the figure

fig.patches.append(arrow)

plt.tight_layout()
plt.show()
