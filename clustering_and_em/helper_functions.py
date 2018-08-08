import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import Math, HTML, display, Latex
import matplotlib
import numpy as np
import time

def dropdown_math(title, text = None, file = None):
    out = widgets.Output()
    with out:
        if not(file == None):
            handle = open(file, 'r')
            display(Latex(handle.read()))
        else:
            display(Math(text))
    accordion = widgets.Accordion(children=[out])
    accordion.set_title(0, title)
    accordion.selected_index = None
    return accordion

def remove_axes(which_axes = '', subplot = None):
    
    frame = plt.gca()
        
    if subplot is None:
        if 'x' in which_axes:
            frame.axes.get_xaxis().set_visible(False)
        if 'y' in which_axes:
            frame.axes.get_yaxis().set_visible(False)

        elif which_axes == '':
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            
    else:
        if subplot[2] % subplot[1] is not 1:
            frame.axes.get_yaxis().set_visible(False)
            
        if ((subplot[2] - 1) // subplot[1]) < subplot[0] - 1:
            frame.axes.get_xaxis().set_visible(False)
    return

def set_notebook_preferences():

    display(HTML("""
    <style>
    .output {
        font-family: "Georgia", serif;
        align-items: normal;
        text-align: normal;
    }
    
    div.output_svg div { margin : auto; }

    .div.output_area.MathJax_Display{ text-align: center; }

    div.text_cell_render { font-family: "Georgia", serif; }
    
    details {
        margin: 20px 0px;
        padding: 0px 10px;
        border-radius: 3px;
        border-style: solid;
        border-color: black;
        border-width: 2px;
    }

    details div{padding: 20px 30px;}

    details summary{font-size: 18px;}
    
    table { margin: auto !important; }
    
    </style>
    """))

    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'Georgia'

    matplotlib.rc('axes', titlesize = 18)
    matplotlib.rc('axes', labelsize = 16)
    matplotlib.rc('xtick', labelsize = 12)
    matplotlib.rc('ytick', labelsize = 12)

def beautify_plot(params):

    if not(params.get('title', None) == None):
        plt.title(params.get('title'))

    if not(params.get('x', None) == None):
        plt.xlabel(params.get('x'))

    if not(params.get('y', None) == None):
        plt.ylabel(params.get('y'))


def sample_weights_from(w1, w2, post):
    idx1, idx2 = np.arange(0, post.shape[0]), np.arange(0, post.shape[1])
    idx1, idx2 = np.meshgrid(idx1, idx2)
    idx = np.stack([idx1, idx2], axis = 2)
    idx = np.reshape(idx, (-1, 2))

    flat_post = np.reshape(post, (-1,)).copy()
    flat_post /= flat_post.sum()

    sample_idx = np.random.choice(np.arange(0, flat_post.shape[0]), p = flat_post)
    grid_idx = idx[sample_idx]

    return w1[grid_idx[0]], w2[grid_idx[1]]



def kNN(X_train, Y_train, X_test, k, p = 2):
    
    X_test_clone = np.stack([X_test]*X_train.shape[0], axis = -2) # clone test points for comparisons as before
    distances = np.sum(np.abs(X_test_clone - X_train)**p, axis = -1) # compute Lp distances
    idx = np.argsort(distances, axis = -1)[:, :k] # find k smallest distances
    classes = Y_train[idx] # classes corresponding to the k smallest distances
    predictions = []
    
    for class_ in classes:
        uniques, counts = np.unique(class_, return_counts = True)
        
        if (counts == counts.max()).sum() == 1:
            predictions.append(uniques[np.argmax(counts)])
        else:
            predictions.append(np.random.choice(uniques[np.where(counts == counts.max())[0]]))
            
    return np.array(predictions)


def sig(x):
    return 1/(1 + np.exp(-x))

def logistic_gradient_ascent(x, y, init_weights, no_steps, stepsize):
    x = np.append(np.ones(shape = (x.shape[0], 1)), x, axis = 1)
    w = init_weights.copy()
    w_history, log_liks = [], []
    
    for n in range(no_steps):
        log_liks.append(np.sum(y*np.log(sig(x.dot(w))) + (1 - y)*np.log(1 - sig(x.dot(w)))))
        w_history.append(w.copy())
    
        sigs = sig(x.dot(w))
        dL_dw = np.mean((y - sigs)*x.T, axis = 1)
        w += stepsize*dL_dw
    
    return np.array(w_history), np.array(log_liks)

def softmax(x):
    return (np.exp(x).T/np.sum(np.exp(x), axis = 1)).T

def softmax_gradient_ascent(x, y, init_weights, no_steps, stepsize):
    x = np.append(np.ones(shape = (x.shape[0], 1)), x, axis = 1)
    w = init_weights.copy()
    w_history, log_liks = [], []

    for n in range(no_steps):
        log_liks.append(np.sum(y*np.log(softmax(x.dot(w)))))
        w_history.append(w.copy())
    
        soft_ = softmax(x.dot(w))
        dL_dw = (x.T).dot(y - soft_)/x.shape[0]
        w += stepsize*dL_dw
    
    return np.array(w_history), np.array(log_liks)

def PCA_N(x):
    
    
    S = ((x - x.mean(axis = 0)).T).dot(x - x.mean(axis = 0))/x.shape[0]
    
    t = time.time()
    eig_values, eig_vectors = np.linalg.eig(S)
    print('Time taken for high-dimensional approach:', np.round((time.time() - t), 3), 'sec')
    
    sort_idx = (-eig_values).argsort()
    eig_values, eig_vectors = eig_values[sort_idx], eig_vectors[:, sort_idx]
    
    return np.real(eig_values), np.real(eig_vectors)

def k_means(x, K, max_steps, mu_init):
    
    N, D = x.shape # N: number of datapoints, D: number of input dimensions
    mus = mu_init.copy() # copy cluster centers to avoid mutation

    s = np.zeros(shape = (N, K)) # set all membership indices to 0
    memberships = np.random.choice(np.arange(0, K), N) # randomly choose a cluster for each point
    s[np.arange(s.shape[0]), memberships] = 1 # set the 1-entries according to the random sample
    
    x_stacked = np.stack([x]*K, axis = 1) # stack K copies of x to do calculation for each cluster
    losses = [np.sum(s*np.sum((x_stacked - mus)**2, axis = 2))] # array to store costs, containing 1st cost
    converged = False
    
    for i in range(max_steps):

        distances = np.sum((x_stacked - mus)**2, axis = 2) # find which cluster mean is closest to each point
        min_idx = np.argmin(distances, axis = 1)
        s_prev = s.copy()
        s = np.zeros_like(s)
        s[np.arange(s.shape[0]), min_idx] = 1
        
        mus = (s.T).dot(x) # compute K centers in one go
        N_k = s.sum(axis = 0).reshape((-1, 1)) # number of members of cluster k
        mus[np.where(N_k >= 1)[0], :] /= N_k[np.where(N_k >= 1)[0]] # distance of each point from each cluster mean

        losses.append(np.sum(s*np.sum((x_stacked - mus)**2, axis = 2)))
        
        if np.prod(np.argmax(s, axis = 1) == np.argmax(s_prev, axis = 1)):
            break
            
    return s, mus, losses
