import numpy as np

np.random.seed(0)
cov = np.array([[1, 0.995],
                [0.995, 1]])

mu = np.array([0, 0])
norm_rands = np.random.multivariate_normal(mu, cov, 100)

np.save('corr_data_2d.npy', norm_rands)
