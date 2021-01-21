import numpy as np
import matplotlib.pyplot as plt

def get_kalman_gain(V_, C_, R_):
    inverse_term = np.linalg.inv(C_.dot(V_).dot(C_.T) + R_)
    
    return V_.dot(C.T).dot(inverse_term)


def kalman_filter(y, A_, Q_, C_, R_, mean_0, cov_0):
    
    K = get_kalman_gain(cov_0, C_, R_) # kalman gain at t = 0
    means_1 = [mean_0 + K.dot(y[0] - C_.dot(mean_0))] # array to store x^(t-1)_t
    covs_1 = [(np.eye(C_.shape[-1]) - K.dot(C_)).dot(cov_0)] # array to store V^(t-1)_t
    means_2, covs_2 = [], [] # arrays to store x^t_t, V^t_t
    
    for t in range(1, len(y)):
        means_2.append(A_.dot(means_1[-1])) # store x^t_(t-1)
        covs_2.append(A_.dot(covs_1[-1]).dot(A_.T) + Q_) # store V^t_(t-1)
        
        K = get_kalman_gain(covs_2[-1], C_, R_) # store x^t_(t-1)
        means_1.append(A_.dot(means_2[-1]) + K.dot(y[t] - C_.dot(means_2[-1]))) # store x^t_t
        covs_1.append((np.eye(C_.shape[-1]) - K.dot(C_)).dot(covs_1[-1])) # store V^t_t
        
    return np.array(means_1), np.array(covs_1), np.array(means_2), np.array(covs_2)

def sample_lds(seq_length, A_, C_, Q_, R_, mu_0_, V_0_):

    xs = [np.random.multivariate_normal(mu_0_, V_0_)]
    ys = [np.random.multivariate_normal(C_.dot(xs[-1]), R_)]
    
    for i in range(seq_length):
        xs.append(np.random.multivariate_normal(A_.dot(xs[-1]), Q_))
        ys.append(np.random.multivariate_normal(C_.dot(xs[-1]), R_))
        
    return xs, ys

np.random.seed(2)
mu_0, V_0 = np.array([0]), np.array([[1]])
lamda = 0.99

##A = np.array([[lamda]])
##Q = np.array([[1 - lamda**2]])
##C = np.array([[2]])
##R = 0.01*np.eye(1)
mu_0, V_0 = np.array([1, 0]), np.array([[1, 0], [0, 1]]) # initial latent state parameters
lamda, theta = 0.99, 2*np.pi/100 # constants used later to express A, Q matrices

A = lamda*np.array([[np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)]])
Q = (1 - lamda**2)*np.eye(2)
C = np.array([[1, 0]])
R = np.array([[0.01]])
no_points = 100
xs, ys = sample_lds(no_points, A, C, Q, R, mu_0, V_0)

x_1, V_1, x_2, V_2 = kalman_filter(ys, A, Q, C, R, mu_0, V_0)

##mean = C.dot(A).dot(x_1.T).T[:-1, 0]
##std = C.dot(A).dot(V_1.dot(A.T).dot(C.T)) + C.dot(Q).dot(C.T) + R
##print(std.shape)
##std = std[0, :-1, 0]**0.5

mean = C.dot(x_2.T).T[:, 0]
std = C.dot(V_2).dot(C.T)[0, :, 0]**0.5

std_x = V_2[:, 0, 0]

print(mean.shape, std.shape)
x_lim = [-1, 21]
y_lim = [-2, 2]

for i in range(21):

    # new y
    plt.subplot(1, 2, 1)
    
    if i > 0:
        plt.errorbar(np.arange(i), mean[:i], xerr = 0, yerr = std[:i],
                     fmt = None, color = 'grey')
        plt.errorbar(i, mean[i], xerr = 0, yerr = std[i], fmt = None,
                     color = 'black')
    plt.scatter(np.arange(i+1), ys[:i+1], marker = 'x', s = 25, color = 'black')
    plt.xlim(x_lim), plt.ylim(y_lim)

    plt.subplot(1, 2, 2)
    if i > 0:
        plt.errorbar(np.arange(i), x_2[:i, 0], xerr = 0, yerr = std_x[:i],
                     fmt = None, color = 'grey')
    plt.xlim(x_lim)
    plt.show()

    # x pred

    # y pred

