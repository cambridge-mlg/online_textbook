# 2. Regression

In the previous section we used the method \\(\texttt{stats.linregress}\\) to perform least squares regression quite liberally and without explaining what the function does under the hood. We also deferred questions like our definition of error and lack of error bars. In this section we will look at regression in detail to address these issues. The term regression describes a broad class of problems where the aim is to predict a **continuous output** \\(y^\star\\) given its input \\(x^\star\\), and a set of example \\(\{x_n, y_n\}\\) pairs. Such problems amount to finding a function \\(y = f(x)\\) which describes the data sufficiently well,  allowing us to make future predictions \\(y^\star = f(x^\star)\\).

Many real tasks such as temperature forecasts, sales and stock price predictions are regression problems making regression methods relevant and very applicable. More importantly, regression serves as a good introduction to several recurrent concepts in inference and machine learning like *generative models*, *maximum likelihood approaches*, *overfitting* and *bayesian inference*. It is highly recommended that you take time to grasp the concepts in this section well \\(-\\) it will pay off later.

%config InlineBackend.figure_format = 'svg'
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *

set_notebook_preferences()

## 2.1 Linear regression

Let's start by looking at the details of linear fits using a prepared toy dataset, starting by plotting the dataset.

x_lin = np.load('reg_lin_x.npy') # np.load loads a prepared numpy array from the file 'reg_line_x.npy'
y_lin = np.load('reg_lin_y.npy') # same for 'reg_line_y.npy'

plt.scatter(x_lin, y_lin, marker = 'x', color = 'red') # scatter plot of the inputs, x, and outputs, y
beautify_plot({"title":"Linear dataset", "x":"$x$", "y":"$y$"})
plt.show() # show the scatter plot

We now want to find the straight line with slope $w_1$ and intercept $w_0$ which describes the data.

\begin{align}
y = w_1 x + w_0
\end{align}

A straight line clearly can't pass through all the datapoints because they aren't collinear so we must define what a good description of the data is. In the previous section we presented the sum-of-squares error as a candidate definition

\begin{align}
E_2 = \sum^N_{n = 1} \big[y_n - (w_1x_n + w_0)\big]^2 \geq 0
\end{align}

where equality with 0 holds only if \\(~y_n = w_1x_n + w_0\\) for every \\(n\\). We then define the optimal fit as that which minimises \\(E_2\\) $-$ and proceed by finding the values of \\(w_0, w_1\\) which do so. However why minimize the sum-of-squares instead of 

\begin{align}
E_{p} = \sum^N_{n = 1} \big|y_n - (w_1x_n + w_0)\big|^p,~\text{for some $p > 0$}?
\end{align}

For the moment we'll stick with \\(p = 2\\) and there are two good reasons for this:

- Minimizing \\(E_2\\) makes the maths easier and gives a closed-form solution for the \\(w\\)'s.

- It turns out that minimising \\(E_2\\) is equivalent to finding the \\(w\\)'s which are most likely, given the data, if we assume that some gaussian measurement noise has been added to each \\(y_n\\). This not only makes the use of \\(E_2\\) interpretable, but is the method one should employ for gaussian-noisy data.

We will demonstrate both these points shortly. Skipping ahead slightly, let's have a look at the following animation to get a feel for the problem. In the left plot, we have visualised the line in data space for different \\((w_0, w_1)\\) values and shown the (unsquared) errors in gray dashes. The right plot, is a contour of \\(\text{ln}(E_2)\\) in the weight space \\(-\\) blue means low and red means high log-error. The black cross shows the \\((w_0, w_1)\\) pair corresponding to the displayed line.

<div class="row">
  <div class="column">
    <img src="reg_lin_weight_excursion.gif" style="width:80%; float: center; padding: 0px">
  </div>
</div>


Our goal is to find the exact \\((w_0, w_1)\\) which minimise the error. We proceed by writing \\(E_2\\) in the more convenient notation

\\[
E_2 = \big|\mathbf{y} - \mathbf{X}\mathbf{w}\big|^2 = \big(\mathbf{y} - \mathbf{X}\mathbf{w}\big)^\top \big(\mathbf{y} - \mathbf{X}\mathbf{w}\big)
\\]


\begin{equation}
\text{where}~~~
\mathbf{y} = \begin{pmatrix}
y_1\\\
y_2\\\
\vdots \\\
y_N\\\
\end{pmatrix}, ~~~
\mathbf{X} =  \begin{pmatrix}
1 & x_1\\\
1 & x_2\\\
\vdots & \vdots \\\
1 & x_N\\\
\end{pmatrix}, ~~~
\mathbf{w} =  \begin{pmatrix}
w_0\\\
w_1\\\
\end{pmatrix}
\end{equation}

<details>
<summary>Index notation in detail</summary>
<div>
\begin{align}
    E_2 &= \sum^N_{n = 1} \big[y_n - (w_1x_n + w_0)\big]^2\\
    ~\\
    &= \sum^N_{n = 1} \big[\mathbf{y}_n - \sum^2_{j = 1}\mathbf{X}_{nj}\mathbf{w}_j\big]^2\\
    ~\\
    &= \sum^N_{n = 1} \big[\mathbf{y}_n - \left(\mathbf{X}\mathbf{w}\right)_n\big]^2\\
    ~\\
    &= \big(\mathbf{y} - \mathbf{X}\mathbf{w}\big)^\top \big(\mathbf{y} - \mathbf{X}\mathbf{w}\big)\\
\end{align}
</div>
</details>

Also, introducing the notation for the derivative of a quantity \\(f\\) with respect to a vector \\(\mathbf{v}\\)

\\[
\bigg(\frac{\partial f}{\partial \mathbf{v}}\bigg)_i = \frac{\partial f}{\partial \mathbf{v}_i}
\\]

and using it to extremize \\(E_2\\), we obtain the closed form solution:

\begin{align}\frac{\partial E_2}{\partial \mathbf{w}} &= -2\mathbf{X}^\top\big(\mathbf{y} - \mathbf{X}\mathbf{w}\big)=0\\
~\\\
\implies & \mathbf{X}^\top\mathbf{X}\mathbf{w} - \mathbf{X}^\top\mathbf{y} = 0\\
~\\\
\implies &\boxed{\mathbf{w} = \big( \mathbf{X}^\top\mathbf{X}\big)^{-1}\mathbf{X}^\top \mathbf{y}}
\end{align}

<details>
<summary>Derivatives in detail</summary>
<div>
Here we show a more detailed derivation of the equality $\frac{\partial E_2}{\partial \mathbf{w}} = -2\mathbf{X}^\top\big(\mathbf{y} - \mathbf{X}\mathbf{w}\big)$ in case the vector notation of derivatives is not clear.

\begin{align}
\bigg(\frac{\partial E_2}{\partial \mathbf{w}}\bigg)_i &= \frac{\partial E_2}{\partial \mathbf{w}_i} = \frac{\partial}{\partial \mathbf{w}_i} \bigg[\big(\mathbf{y} - \mathbf{X}\mathbf{w}\big)^\top \big(\mathbf{y} - \mathbf{X}\mathbf{w}\big)\bigg] = \frac{\partial}{\partial \mathbf{w}_i} \sum_n \bigg[\big(\mathbf{y}_n - \sum_j\mathbf{X}_{nj}\mathbf{w}_j\big) \big(\mathbf{y}_n - \sum_j\mathbf{X}_{nj}\mathbf{w}_j\big)\bigg]\\
~\\
&= 2\sum_n \bigg[\big(\mathbf{y}_n - \sum_j\mathbf{X}_{nj}\mathbf{w}_j\big) \frac{\partial}{\partial \mathbf{w}_i} \big(\mathbf{y}_n - \sum_j\mathbf{X}_{nj}\mathbf{w}_j\big)\bigg]\\
~\\
&= -2\sum_n \bigg[\big(\mathbf{y}_n - \sum_j\mathbf{X}_{nj}\mathbf{w}_j\big) \big(\sum_j\mathbf{X}_{nj} \frac{\partial \mathbf{w}_j}{\partial \mathbf{w}_i}\big)\bigg]\\
~\\
&= -2\sum_n \bigg[\big(\mathbf{y}_n - \sum_j\mathbf{X}_{nj}\mathbf{w}_j\big) \big(\sum_j\mathbf{X}_{nj} \delta_{ij}\big)\bigg]\\
~\\
&= -2\sum_n \bigg[\big(\mathbf{y}_n - \sum_j\mathbf{X}_{nj}\mathbf{w}_j\big)\mathbf{X}_{ni}\bigg]\\
~\\
&= -2\sum_n \bigg[\mathbf{X}^\top_{in}\big(\mathbf{y}_n - \sum_j\mathbf{X}_{nj}\mathbf{w}_j\big)\bigg]\\
~\\
&= -2 \left[\mathbf{X}^\top \big(\mathbf{y} - \mathbf{X}\mathbf{w}\big)\right]_i\\
\end{align}
</div>
</details>

As promised, we have a closed form solution for \\(\mathbf{w}\\) which extremizes \\(E_2\\). You can convince yourself this is a minimum either by taking a second derivative or by first considering the quadratic form of \\(E_2\\) and then what happens in the limit of large \\(\mathbf{w}\\). In this expression for \\(\mathbf{w}\\)

$$\mathbf{w} = \big( \mathbf{X}^\top\mathbf{X}\big)^{-1}\mathbf{X}^\top \mathbf{y}$$

it is interesting to note that the matrix $\big( \mathbf{X}^\top\mathbf{X}\big)^{-1}\mathbf{X}^\top$ is a generalization of the inverse of a matrix for non-square matrices, called the __[Moore-Penrose pseudoinverse](http://mathworld.wolfram.com/Moore-PenroseMatrixInverse.html)__:

$$\bigg[\big( \mathbf{X}^\top\mathbf{X}\big)^{-1}\mathbf{X}^\top\bigg] \mathbf{X} = \big( \mathbf{X}^\top\mathbf{X}\big)^{-1}\mathbf{X}^\top\mathbf{X} = \mathbf{I}$$

Implementing this solution is straightforward because of the closed-form solution we have:

ones = np.ones_like(x_lin) # create a vector of 1's with the same length as x
X = np.stack([ones, x_lin], axis = 1) # stack 1's and x's to get the X matrix having the 1's and x's as columns

w = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(y_lin) # compute the optimal w using the Moore-Penrose pseudoinverse

x_pred = np.linspace(0, 1, 100) # 100 points equispaced between 0 and 1
y_pred = w[1]*x_pred + w[0] # evaluate the linear trendline at the values of x above

plt.plot(x_pred, y_pred, color = 'black') # plot the trendline
plt.scatter(x_lin, y_lin, marker = 'x', color = 'red') # plot the datapoints
beautify_plot({"title":"Least squares regression", "x":"$x$", "y":"$y$"})
plt.show()

Now that we've demonstrated the least-squares solution has a tractable closed form, we turn our attention to the interesting equivalence advertised previously: that minimizing \\(E_2\\) is equivalent to determining the most likely w's under the assumption the \\(y\\)'s contain gaussian noise. Suppose the \\(y\\)'s have been generated through

\\[y_n = w_0 + w_1 x_n + \epsilon_n,~\text{where}~ \epsilon_n \sim \mathcal{N}(0, \sigma^2)\\]

where we have added *independent* gaussian noise \\(\epsilon_n\\) to each true value of \\(w_0 + w_1 x_n\\). By independent we mean that for each data point we have drawn a different noise value \\(\epsilon_n\\). For given \\(\sigma^2,\\) \\(\mathbf{X}\\) and \\(\mathbf{w}\\), we can write down the probability density of \\(\mathbf{y}\\) or likelihood as

\\[p(\mathbf{y}\mid\mathbf{X}, \mathbf{w}, \sigma^2) = \frac{1}{(2\pi \sigma^2)^{N/2}}\text{exp}\big(-\frac{1}{2\sigma^2}(\mathbf{y} - \mathbf{X}\mathbf{w})^\top (\mathbf{y} - \mathbf{X}\mathbf{w})\big)\\]

Now we aim to find the \\(\mathbf{w}\\) which maximizes the likelihood. Instead of directly taking derivatives of the exponential, we can use the simplifying facts: (i) the maxima/minima of a quantity are also maxima/minima of a monotonic increasing function \\(f\\) (monotonic increasing means \\(f\\) decreases with \\(x\\)) of that quantity and that (ii) the logarithm is a monotonic increasing function. Then maximixing \\(p(\mathbf{y}\mid\mathbf{X}, \mathbf{w}, \sigma^2)\\) is equivalent to maximizing the **log-likelihood**

\\[\mathcal{L} = \text{log}~ p(\mathbf{y}\mid\mathbf{X}, \mathbf{w}, \sigma^2) = -\frac{N}{2}log(2\pi \sigma^2) -\frac{1}{2\sigma^2}(\mathbf{y} - \mathbf{X}\mathbf{w})^\top (\mathbf{y} - \mathbf{X}\mathbf{w})\\]

or equivalently **minimizing** the negative log-likelihood

\\[-\mathcal{L} = \frac{N}{2}log(2\pi \sigma^2) +\frac{1}{2\sigma^2}(\mathbf{y} - \mathbf{X}\mathbf{w})^\top (\mathbf{y} - \mathbf{X}\mathbf{w})\\]

<details>
<summary>Extra proof on extrema of monotonic functions</summary>
<div>
Consider a quantity \\(\mathcal{Q}(x)\\) with an extremum at \\(x^*\\), and a monotonic function \\(f\\). The derivative of \\(f(\mathcal{Q}(x))\\) with respect to \\(x\\) is

\begin{align}
\frac{\partial~ f\big(\mathcal{Q}\big)}{\partial x} = \frac{\partial~ f}{\partial \mathcal{Q}} \frac{\partial~ \mathcal{Q}}{\partial x}.
\end{align}

So at \\(x = x^*\\), \\(\partial~ f\left(\mathcal{Q}\right)/\partial x = 0\\) and also the signs of \\(\partial^2~ f\big(\mathcal{Q}\big)/\partial x^2\\) and \\(\partial^2~ \mathcal{Q}/\partial x^2\\) are the same, which means that the type of extremum is the same for \\(\mathcal{Q}\\) and \\(f\big(\mathcal{Q}\big)\\): the maxima of \\(\mathcal{Q}\\) are maxima of \\(f(\mathcal{Q})\\) and similarly for the minima.
</div>
</details>

Because the term \\(\frac{N}{2}log(2\pi \sigma^2)\\) is independent of \\(\mathbf{w}\\), minimizing the negative log-likelihood is equivalent to minimizing the least-squares error \\(-\\) exactly the same criterion we had before:

\\[\boxed{\text{Least squares} \equiv \text{minimize}~ (\mathbf{y} - \mathbf{X}\mathbf{w})^\top (\mathbf{y} - \mathbf{X}\mathbf{w}) \Leftrightarrow \text{Maximum-likelihood}}\\]

As promised, we have shown the advertised equivalence. Aside from providing an interpretation for least squares, our probabilistic approach has several additional benefits which we will exploit later. Our probabilistic treatment will allow us to infer the noise level, \\(\sigma\\), as well as quantitatively criticise our model. More importantly however, this probabilistic approach paves the way to Bayesian modelling where the task of selecting model weights is dealt with in a principled manner.

## 2.2 Non-linear basis regression

In the previous section we showed how linear regression can be used to deal with datasets where the input-output relation is linear. We can extend our method to deal with non-linear datasets with minimal effort, by modelling the dataset as a linear combination of basis functions \\(\phi_d, d = 0, 1, ..., D\\). For example we may choose a set of polynomials or sinusoids

\\[\phi_{d} = x^d, ~\text{or}~
\phi_{d} = e^{i\pi d},\\]

or any other set of functions \\(\phi_{d}\\). Then the assumed model responsible for generating the data, called the **generative** model is:


\\[y_n = w_0 + w_1 \phi_{1}(x_n) + w_2 \phi_{2}(x_n) + ... w_D \phi_{D}(x_n) + \epsilon_n = \boldsymbol{\phi}(x_n)^\top \mathbf{w} + \epsilon_n\\]

Note that the set of functions \\(\phi_{d}\\) does not need to be a complete basis \\(-\\) a set of functions is complete if every function can be expressed as a linear combination of functions from the set. In fact we cannot possibly use a complete set in practice because we have to cut off the sum at some point, or else we would need infinite computation time! Using the same approach as before, we write

\\[\mathbf{y} = \boldsymbol{\Phi}\mathbf{w} + \boldsymbol{\epsilon}\\]

where \\(\mathbf{y}\\) is the vector of outputs, \\(\boldsymbol{\epsilon}\\) is a vector of independent gaussian noise draws from \\(\mathcal{N}(0, \sigma^2)\\) and \\(\boldsymbol{\Phi}\\) is the matrix whose entry at the \\(i^{th}\\) row and \\(j^{th}\\) column is \\(\phi_j(x_i)\\). The matrix \\(\boldsymbol{\Phi}\\) is called the **design matrix** and when written out, it looks like:


\begin{equation}
\boldsymbol{\Phi} =  \begin{pmatrix}
1 & \phi_1(x_1) & \cdots & \phi_D(x_1)\\\
1 & \phi_1(x_2) & \cdots & \phi_D(x_2)\\\
\vdots & \vdots & \ddots & \vdots \\\
1 & \phi_1(x_N) & \cdots & \phi_D(x_N)\\\
\end{pmatrix}
\end{equation}

Where we have explicitly assumed the \\(0^{th}\\) basis function is 1 to give the constant \\(w_0\\) term when multiplied by \\(\mathbf{w}\\). You should convinve yourself that this matrix gives the correct linear combination when acting on \\(\mathbf{w}\\). We can now proceed either by doing least squares with error

\\[E_2 = \big|\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}\big|^2 = \big(\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}\big)^\top \big(\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}\big),\\]

or by minimizing the negative log-likelihood

\\[- \mathcal{L} = - \text{log}~ p(\mathbf{y}|\boldsymbol{\Phi}, \mathbf{w}, \sigma^2) = \frac{N}{2}log(2\pi \sigma^2) + \frac{1}{2\sigma^2}(\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})^\top (\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})\\]

with respect to $\mathbf{w}$. Again it is easy to convince yourself that the two approaches are equivalent, and that the maximum-likelihood weights are

\begin{align}
\boxed{\mathbf{w} = \big( \boldsymbol{\Phi}^\top\boldsymbol{\Phi}\big)^{-1}\boldsymbol{\Phi}^\top \mathbf{y}}
\end{align}

Implementing the solution is once again straightforward:

x_nonlin = np.load('reg_nonlin_x.npy') # load inputs from a prepared non-linear dataset
y_nonlin = np.load('reg_nonlin_y.npy') # load corresponding outputs

plt.scatter(x_nonlin, y_nonlin, marker = 'x', color = 'red')
beautify_plot({"title":"Non-linear dataset", "x":"$x$", "y":"$y$"})
plt.show()

For this example we will use polynomial basis functions \\(\phi_d(x) = x^d\\). The next question is where to cut off the linear combination. Intuitively, we expect a higher order polynomial to be more flexible and able to model more complex input-output relations. For the momement, let's arbitrarily pick \\(D = 3\\).

D = 3
phi = np.array([[x_**d for d in range(0, D + 1)] for x_ in x_nonlin]) 
w = np.linalg.inv((phi.T).dot(phi)).dot(phi.T).dot(y_nonlin) # apply the Moore-Penrose pseudoinverse using Phi

xs = np.linspace(0, 1, 100) # 100 points equispaced between 0 and 1
phi_pred = np.array([[x_**d for d in range(0, D + 1)] for x_ in xs]) # design matrix for points at which to plot
ys = phi_pred.dot(w) # output of the model at the points above

print('Sum squared errors for polynomial of order {}:'.format(D), np.sum((phi.dot(w)-y_nonlin)**2).round(3))
plt.scatter(x_nonlin, y_nonlin, marker = 'x', color = 'red') # plot model predictions
plt.plot(xs, ys, color = 'black') # plot dataset
beautify_plot({"title":"Non-linear regression (D = {})".format(D), "x":"x", "y":"y"})
plt.show()

How does then the wellness of fit depend on the order of the polynomial? In the code above, **experiment with $D$** and see how the sum of squares of error changes. Let's repeat the process for $D = 1, 2, ... 9$.

fig = plt.figure(figsize = (9, 9)) # figure on which to plot the subfigures - you don't have to worry about this
for D in range(1, 10):
    phi = np.array([[x_**d for d in range(0, D + 1)] for x_ in x_nonlin]) # training design matrix, as before
    w = np.linalg.inv((phi.T).dot(phi)).dot(phi.T).dot(y_nonlin) # Moore-Penrose pseudoinverse
    
    phi_pred = np.array([[x_**d for d in range(0, D + 1)] for x_ in xs]) # design matrix of evaluation points as before
    ys = phi_pred.dot(w) # model predictions as before
    
    plt.subplot(3, 3, D)
    plt.scatter(x_nonlin, y_nonlin, marker = 'x', color = 'red') # plot 
    plt.plot(xs, ys, color = 'black')
    plt.xlim([0, 1])
    plt.ylim([-1.5, 1.5])
    if D % 3 is not 1:
        remove_axes('y')
    if D < 7:
        remove_axes('x')
plt.tight_layout()
plt.show()

In the plots above something interesting is happening as \\(D\\) changes. As the order of the polynomial increases, it becomes more flexible and it can pass closer to each datapoint. At \\(D = 9\\), the polynomial has \\(10\\) degrees of freedom (including the constant \\(w_0\\)), which is equal to the number of datapoints and is just enough for the curve to pass exactly through every point achieving an error of 0. In effect, the polynomial is using up its degrees of freedom to fit the training datapoints exessively well, whilst forming an odd shape which we do not expect to represent unseen data very well \\(-\\) notice how the curve becomes excessively wiggly as \\(D\\) increases. This phenomenon is called **overfitting** and is a serious problem which occurs when the model complexity becomes large compared to the amount of training data. Overfitted models exhibit very small training errors but are too well adapted for the training data and *learn the noise* of that data too. Consequently, they make poor predictions about unseen datapoints \\(-\\) they fail to generalise. Returning to our polynomial example, let's have a look at the values of the weights for each \\(D\\).

weights = []
for D in range(0, 10):
    phi = np.array([[x_**d for d in range(0, D + 1)] for x_ in x_nonlin])
    w = np.linalg.inv((phi.T).dot(phi)).dot(phi.T).dot(y_nonlin)
    
    w = np.pad(w, (0, 10 - w.shape[0]), 'constant', constant_values = (0, 0)) # pad with 0's for unused weights
    weights.append(w) 

import pandas
row_names, column_names = [d for d in range(10)], ['$w_{}$'.format(d) for d in range(10)]
table = pandas.DataFrame(weights, row_names, column_names).round(2)
table.columns.name = "D"
table

The weights also show something interesting: as \\(D\\) increases the values of the high-order weights increase dramatically and also their sign oscillates with \\(d\\), i.e. \\(w_d\\) and \\(w_{d+1}\\) have opposite signs for a given large \\(D\\). The sign oscillation is because each (extremely large) term must be counteracted by some other term, so the weights come in pairs of opposite signs. Prior to fitting the model, we would hardly expect such a behaviour of the \\(w\\)'s and something must be done to fix this overfitting problem. But before we look for a solution, let's quantify the impact of overfitting on our model's performance.

How can we evaluate model performance and diagnose overfitting? One way to do this is to use a fraction of the data to train the model (train set), whilst leaving the rest of the data (test set) unseen. We can evaluate the model's performance on the train and test sets. If training performance is good while test performance is poor, it is likely that there is overfitting, whereas if training performance is poor to start with, it is likely that the model is exeedingly simple to capture the complexity of the dataset.

Let's apply this method to our polynomial model, using an extended version of the previous dataset. This contains the 10 points of the previous dataset plus another 40 points which we'll use for testing.

x_ext = np.load('reg_nonlin_x_extended.npy')
y_ext = np.load('reg_nonlin_y_extended.npy')
x_train, x_test, y_train, y_test = x_ext[:10], x_ext[10:], y_ext[:10], y_ext[10:]

train_err, test_err = [], [] # lists to store training and test error as D varies
for D in range(0, 10):
    
    phi = np.array([[x_**d for d in range(0, D + 1)] for x_ in x_train]) # design matrix for training points
    w = np.linalg.inv((phi.T).dot(phi)).dot(phi.T).dot(y_train) # max-lik w
    y_trained = phi.dot(w) # evaluate polynomial at training points
    train_err.append(((y_trained - y_train)**2).mean()**0.5) # store train errors
    
    phi_pred = np.array([[x_**d for d in range(0, D + 1)] for x_ in x_test]) # design matrix for test points
    y_pred = phi_pred.dot(w) # evaluate polynomial at test data points
    test_err.append(((y_pred - y_test)**2).mean()**0.5) # store test errors
    
plt.plot(np.arange(0, 10), train_err, color = 'blue', marker = 'o', label = 'Train')
plt.plot(np.arange(0, 10), test_err, color = 'red', marker = 'o', label = 'Test')
plt.gca().legend()
beautify_plot({"title":"Training and test errors", "x":"$D$", "y": '$E_{rms}$'}) # add a legend for maximum style
plt.show()

As the order of the polynomial increases, the training error steadily decreases due to the model's increased flexibility, whereas the test error initially decreases and then increases again, because the model starts overfitting the training data and no longer fits the test data well. Notably, the error shoots up at $D = 9$ when the oscillation of the polynomial is most acute.

A remedy for overfitting would be to somehow force the weight coefficients to be small, for example by adding a term to the log-likelihood expression which penalizes large weights:

$$\begin{align}\mathcal{L} \to \mathcal{L} - \frac{\alpha}{2}\mathbf{w}^\top\mathbf{w}= -\frac{N}{2}log(2\pi \sigma^2) -\frac{1}{2\sigma^2}(\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})^\top (\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})\end{align} - \frac{\alpha}{2}\mathbf{w}^\top\mathbf{w}$$

The log-likelihood $\mathcal{L}$ is the same as before, except for the quadratic term $\frac{\alpha}{2}\mathbf{w}^\top\mathbf{w}$ where $\alpha > 0$ is a constant. If a weight term becomes large, this quadratic term penalizes the log-likelihood, and $\alpha$ controls the size of the penalty. This method discourages the weights from becoming large and is known as **regularization**. Again, there is some arbitrariness here, in our choice of regularization term $-$ why use $\mathbf{w}^\top\mathbf{w} = ||\mathbf{w}||^2$ and not $||\mathbf{w}||$ or in fact $||\mathbf{w}||^p$ for arbitrary $p$? We could use a regularization term with different $p$, however the $p = 2$ case has two particularly nice features (which happen to be the same reasons we used least squares $E_2$ instad of $E_p$ earlier!):

- It makes the mathematics tractable

- It is equivalent to performing Bayesian linear regression with a gaussian prior (will be explained later) **fix this**

Note that the literature refers to regularization using different $p$'s as $Lp$, so that $L1 \implies ||\mathbf{w}||$, $~L2 \implies ||\mathbf{w}||^2$ and so on. Sticking with the $p = 2$ case we differentiate $\mathcal{L}$

$$\begin{align}\frac{\partial\mathcal{L}}{\partial \mathbf{w}} = -\frac{1}{\sigma^2}\boldsymbol{\Phi}^\top(\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}) - \alpha\mathbf{w} = 0\\
~\\
\boldsymbol{\Phi}^\top\mathbf{y} - \boldsymbol{\Phi}^\top\boldsymbol{\Phi}\mathbf{w} + \lambda\mathbf{I}\mathbf{w} = 0\\
~\\
\implies \boxed{\mathbf{w} = (\boldsymbol{\Phi}^\top\boldsymbol{\Phi} + \lambda\mathbf{I})^{-1}\boldsymbol{\Phi}^\top\mathbf{y}}\\
\end{align}$$

where we have introduced the term $\lambda = \alpha \sigma^2$, which controls the regularization magnitude relative to the noise magnitude. Comparing this with the unregularized expression for $\mathbf{w}$:

$$\begin{align}
\mathbf{w} &= (\boldsymbol{\Phi}^\top\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\top\mathbf{y},~\text{unregularized},\\
~\\
\mathbf{w} &= (\boldsymbol{\Phi}^\top\boldsymbol{\Phi} + \lambda\mathbf{I})^{-1}\boldsymbol{\Phi}^\top\mathbf{y},~\text{regularized},\\
\end{align}$$

we see that the only difference is the added $\lambda\mathbf{I}$ term. Since this is inside the $(\cdot)^{-1}$ matrix inverse we can intuitively see that its effect is to reduce the magnitude of the matrix elements of $(\boldsymbol{\Phi}^\top\boldsymbol{\Phi} + \lambda\mathbf{I})^{-1}$ and hence those of $\mathbf{w}$. Consider for example the limiting case $\boldsymbol{\Phi}^\top\boldsymbol{\Phi} << \lambda\mathbf{I}$. Then $(\boldsymbol{\Phi}^\top\boldsymbol{\Phi} + \lambda\mathbf{I})^{-1} \approx \lambda^{-1}\mathbf{I}$ and increasing $\lambda$ results in smaller $\mathbf{w}$. Note that the value of $\lambda$ is arbitrarily chosen (we will later address this point). Let's have a look on how regularization affects the model:

fig = plt.figure(figsize = (9, 9))
lamda = 10**-5 # arbitrarily chosen lamda

for D in range(1, 10):
    phi = np.array([[x_**d for d in range(0, D + 1)] for x_ in x_nonlin])
    
    reg_term = lamda*np.eye(D + 1) # regularization term = lamda*(indentity matrix)
    w = np.linalg.inv((phi.T).dot(phi) + reg_term).dot(phi.T).dot(y_nonlin) # apply regularized pseudoinverse
    
    phi_pred = np.array([[x_**d for d in range(0, D + 1)] for x_ in xs]) # design matrix for predictions
    ys = phi_pred.dot(w) # model predictions as before
    
    plt.subplot(3, 3, D)
    plt.scatter(x_nonlin, y_nonlin, marker = 'x', color = 'red')
    plt.plot(xs, ys, color = 'black')
    plt.xlim([0, 1])
    plt.ylim([-1.5, 1.5])
    if D % 3 is not 1:
        remove_axes('y')
    if D < 7:
        remove_axes('x')
plt.tight_layout()
plt.show()

In the plots above, the effect of overfitting is greatly diminished. Now, although regularization appears to deal with overfitting, one could object that we have merely shifted the problem from the choice of model complexity (e.g. where to cut off the polynomial) to the choice of regularization constant \\(\lambda\\). We arbitrarily chose \\(\lambda = 10^{-5}\\) \\(-\\) **try tweaking the value of \\(\boldsymbol{\lambda}\\) to see how the curves change**. One principled way to choose \\(\lambda\\) would be to try different values for it using training/test datasets, and pick the \\(\lambda\\) which results in the best test performance. Let's also have a look on how regularization affects the train/test errors.

x_train, x_test, y_train, y_test = x_ext[:10], x_ext[10:], y_ext[:10], y_ext[10:]
lamda = 10**-5

train_errors, test_errors = [], []
for D in range(0, 10):
    
    reg_term = lamda*np.eye(D + 1)
    phi = np.array([[x_**d for d in range(0, D + 1)] for x_ in x_train])
    w = np.linalg.inv((phi.T).dot(phi) + reg_term).dot(phi.T).dot(y_train)
    y_trained = phi.dot(w)
    train_rms_error = ((y_trained - y_train)**2).mean()**0.5
    
    phi_pred = np.array([[x_**d for d in range(0, D + 1)] for x_ in x_test])
    y_pred = phi_pred.dot(w)
    test_rms_error = ((y_pred - y_test)**2).mean()**0.5
    
    train_errors.append(train_rms_error)
    test_errors.append(test_rms_error)
    
plt.plot(np.arange(0, 10), train_errors, color = 'blue', marker = 'o', label = 'Train')
plt.plot(np.arange(0, 10), test_errors, color = 'red', marker = 'o', label = 'Test')
plt.gca().legend()
beautify_plot({"title":"Training and test errors (regularised)", "x":"$D$", "y": '$E_{rms}$'})
plt.show()

As expected, the training error steadily decreases with $D$. The test error again reaches a minimum for $D = 3$, but unlike in the unregularized case, it doesn't explode for large $D$, because the regularization term prevents the weights from becoming large. To prove this point, let's check the values of the weights:

weights = []
lamda = 10**-6

for D in range(0, 10):
    phi = np.array([[x_**d for d in range(0, D + 1)] for x_ in x_nonlin])
    reg_term = lamda*np.eye(D + 1)
    w = np.linalg.inv((phi.T).dot(phi) + reg_term).dot(phi.T).dot(y_nonlin)
    
    w = np.pad(w, (0, 10 - w.shape[0]), 'constant', constant_values = (0, 0)) # pad with 0's for unused weights
    weights.append(w) 

import pandas
row_names, column_names = [d for d in range(10)], ['$w_{}$'.format(d) for d in range(10)]
table = pandas.DataFrame(weights, row_names, column_names).round(2)
table.columns.name = "D"
table

The weights are significantly decreased by regularization. **You can change $\boldsymbol{\lambda}$ to see how the weights are affected by the size of the regularization term**.

## 2.3 Bayesian linear regression
In the previous section we saw that using a maximum likelihood approach to modelling data can result in overfitting, impairing the model's ability to generalize to unseen data. This occured because the model uses its exessive flexibility to fit the noisy training data exactly. The weights $\mathbf{w}$ attained extremely large values with of alternating signs, which perhaps one would not initially expect. This idea of *expectation* about the weights' values points us towards a Bayesian treatment of linear regression: we will use a distribution to encode our expectations about the values of the weights before observing the data, called the *prior*.

Suppose we introduce a prior over the weights, which we arbitrarily choose to be a gaussian $p(\mathbf{w}) \sim \mathcal{N}(\mathbf{m}_0, \mathbf{S}_0)$, where $\mathbf{m}_0$ is the mean (vector) and $\mathbf{S}_N$ is the covariance (matrix). In full detail, $p(\mathbf{w})$ is

$$\begin{align}
p(\mathbf{w}) = \frac{1}{(2\pi |\mathbf{S}_0|)^{D/2}}\text{exp}\big(-\frac{1}{2\sigma^2}(\mathbf{w} - \mathbf{m}_0)^\top \mathbf{S}_0^{-1} (\mathbf{w} - \mathbf{m}_0)\big),
\end{align}$$

where $D$ is the size of $\mathbf{w}$. Introducing a gaussian prior may appear quite arbitrary $-$ why not use a uniform distribution or any other distribution for that matter? There are two reasons for this

- A gaussian prior makes the mathematics tractable $-$ we'll be able to write down various closed form expressions

- As previously claimed, using a gaussian prior is strongly related to using a quadratic regularization term

Using Bayes' rule, we can relate the prior and likelihood to the posterior:

$$\begin{align}
p(\mathbf{w}|\mathbf{y}, \mathbf{X}, \sigma^2) \propto p(\mathbf{y}|\mathbf{X}, \mathbf{w}, \sigma^2)p(\mathbf{w})
\end{align}$$

up to a normalization constant which we ignore for the moment. Substituting our expressions for the prior and likelihood

$$\begin{align}
p(\mathbf{y}|\mathbf{X}, \mathbf{w}, \sigma^2) &= \frac{1}{(2\pi \sigma^2)^{N/2}}\text{exp}\big(-\frac{1}{2\sigma^2}(\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})^\top (\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})\big)\\
~\\
\implies p(\mathbf{w}|\mathbf{y}, \mathbf{X}, \sigma^2) &\propto \text{exp}\big(-\frac{1}{2\sigma^2}(\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})^\top (\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}) -\frac{1}{2}(\mathbf{w} - \mathbf{m}_0)^\top \mathbf{S}_0^{-1} (\mathbf{w} - \mathbf{m}_0) \big)
\end{align}$$

Next we look to simplify this expression, and we can save ourselves a lot of effort by noting that the exponent is overall quadratic in $\mathbf{w}$, so the expression is a gaussian in $\mathbf{w}$, which is fully determined by its mean and covariance. The exponent of a general multivariate gaussian is a quadratic form

$$\begin{align}
Q = -\frac{1}{2}(\mathbf{w} - \boldsymbol{\mu})^\top \mathbf{S}^{-1} (\mathbf{w} - \boldsymbol{\mu})
\end{align}$$

and the task is to find ($\boldsymbol{\mu}$, $\mathbf{S}^{-1}$) such that

$$
Q = -\frac{1}{2}(\mathbf{w} - \boldsymbol{\mu})^\top \mathbf{S}^{-1} (\mathbf{w} - \boldsymbol{\mu}) =
-\frac{1}{2\sigma^2}(\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})^\top (\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}) -\frac{1}{2}(\mathbf{w} - \mathbf{m}_0)^\top \mathbf{S}_0^{-1} (\mathbf{w} - \mathbf{m}_0).
$$

The covariance of the posterior, $\mathbf{S}^{-1}$, is straightforward to obtain once we notice that it can be read off the term which is quadratic in $\mathbf{w}$:

$$
\mathbf{w}^\top \mathbf{S}^{-1} \mathbf{w} =
\mathbf{w}^\top (\sigma^{-2}\boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \mathbf{S}_0^{-1})\mathbf{w}\\
~\\
\implies \mathbf{S}^{-1} = \sigma^{-2}\boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \mathbf{S}_0^{-1}
$$

Similarly, the mean $\boldsymbol{\mu}$ is straightforward to obtain by noting that it corresponds to the terms linear in $\mathbf{w}$:

$$\begin{align}
\mathbf{w}^\top \mathbf{S}^{-1} \boldsymbol{\mu} &= \mathbf{w}^\top \boldsymbol{\Phi}^\top \mathbf{y} + \mathbf{w}^\top (\mathbf{S}_0^{-1})\mathbf{m}_0\\
~\\
\implies \boldsymbol{\mu} &= \mathbf{S}(\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})^\top (\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})\\
\end{align}$$

There is still one missing piece, namely the value of $\sigma^{-2}$. We can estimate this by calculating the value of $\sigma^{2}$ which maximizes the log-likelihood $\mathcal{L}$:

$$\begin{align}
\frac{\partial\mathcal{L}}{\partial \sigma^{-2}} = -\frac{N}{2\sigma^2} -\frac{1}{2}(\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})^\top (\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}) = 0\\
~\\
\implies \sigma^{2} = \frac{1}{N}(\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})^\top (\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})
\end{align}$$

If we use a zero mean ($\mathbf{m}_0 = 0$) isotropic ($\mathbf{S}_0^{-1} = \lambda \mathbf{I}$) gaussian prior, the posterior mean and covariance become:

$$\begin{align}
\sigma^{2} &= \frac{1}{N}(\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})^\top (\mathbf{y} - \boldsymbol{\Phi}\mathbf{w})\\
~\\
\mathbf{S} &= (\sigma^{-2}\boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \lambda \mathbf{I})^{-1}\\
~\\
\boldsymbol{\mu} &= \mathbf{S}\boldsymbol{\Phi}^\top \mathbf{y}\\
\end{align}$$

This may look familiar:

$$
\boldsymbol{\mu} = (\sigma^{-2}\boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \lambda \mathbf{I})\boldsymbol{\Phi}^\top \mathbf{y} = \text{expression for}~\mathbf{w}~\text{using least-squares with regularization}
$$

Noting that the posterior is maximum when $\mathbf{w} = \boldsymbol{\mu}$ (since a multivariate gaussian attains its maximum at its mean), we conclude that for regression with a gaussian likelihood **finding the $\mathbf{w}$ which maximizes the posterior using a gaussian prior is equivalent to doing least-squares with $\mathbf{L2}$ regularization**. The method whereby we estimate $\mathbf{w}$ as the weight vector which maximizes the posterior is called *maximum a posteriori* or MAP for short.

$$
\boxed{\text{Least squares with } L2 \text{ regularzation } \equiv \text{ MAP with gaussian likelihood and prior}}
$$

Note also that for a different problem the likelihood may not be gaussian but have some other form, and in such cases we may choose another prior to match that likelihood, aiming to make the mathematics tractable. Priors selected to match the form of the likelihood are known as **conjugate priors**.

At this point it is worth reflecting over the Bayesian approach we have employed thus far. Making an assumption about the generative model has allowed us to write down the likelihood $p(\mathbf{y}|\mathbf{X}, \mathbf{w}, \sigma^2)$. Then, assuming the a gaussian prior $p(\mathbf{w})$ over the weights and using Bayes' rule we have determined the (gaussian) posterior distribution $p(\mathbf{w}|\mathbf{y}, \mathbf{X}, \sigma^2)$, and showed that MAP is equivalent to maximum likelihood with $L2$ regularization. Our approach to treating $\mathbf{w}$ however is still one of working out a point estimate and we have so far neglected the overall posterior distribution $p(\mathbf{w}|\mathbf{y}, \mathbf{X}, \sigma^2)$. In fact, we can exploit the posterior further to find the probability distribution of the output $y^*$, given its input $x^*$ and the training data $\mathcal{D}$ through:

$$
p(y^* | x^*, \mathcal{D}) = \int p(y^* | x^*, \mathbf{w}) p(\mathbf{w}|\mathbf{y}, \mathcal{D}) d\mathbf{w} 
$$

where the integral is a multidimensional integral over each weight in $\mathbf{w}$. Evaluating this integral directly can be tedious, but fortunately the calculation can be greatly simplified by noting that since $p(y^* | x^*, \mathbf{w})$ and $p(\mathbf{w}|\mathbf{y}, \mathcal{D})$ are both gaussian, their product is also gaussian and after integrating over $\mathbf{w}$ the result will be a gaussian as well $-$ **take some time to convince yourself that the above is true**. As explained when we calculated the posterior, a multivariate gaussian is fully characterized by its mean and covariance matrix, which we are now after. **You should also convince yourslef** that drawing sample $y^*$'s from $p(y^* | x^*, \mathcal{D})$ is equivalent to drawing samples from $y^* = \boldsymbol{\phi}^\top \mathbf{w} + \epsilon$, where $\mathbf{w} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{S})$ and $\epsilon \sim \mathcal{N}(0, \sigma^2)$. The expectation and variance of $y^*$ are then:


\begin{align}
\mathbb{E}[y^*] &= \mathbb{E}[\boldsymbol{\phi}^\top \mathbf{w} + \epsilon] = \boldsymbol{\phi}^\top\mathbb{E}[\mathbf{w}] = \boldsymbol{\phi}^\top \mu\\
~\\
\text{Var}(y^*) = \mathbb{E}[(y^* - \mathbb{E}[y^*])^2] &= \mathbb{E}[(\boldsymbol{\phi}^\top \mathbf{w} - \boldsymbol{\phi}^\top \mu + \epsilon)^2] = \mathbb{E}[(\boldsymbol{\phi}^\top (\mathbf{w} - \boldsymbol{\mu}) + \epsilon)^2] \\
~\\
&= \boldsymbol{\phi}^\top \mathbb{E}[(\mathbf{w} - \boldsymbol{\mu})(\mathbf{w} - \boldsymbol{\mu})^\top] \boldsymbol{\phi}  + \sigma^2\\
~\\
&= \boldsymbol{\phi}^\top \mathbf{S} \boldsymbol{\phi}  + \sigma^2
\end{align}

$$
\boxed{\mathbb{E}[y^*] = \boldsymbol{\phi}^\top \mu}\\
\boxed{\text{Var}(y^*) = \boldsymbol{\phi}^\top \mathbf{S} \boldsymbol{\phi}  + \sigma^2}
$$

Note that $\mathbb{E}[y^*]$ and $\text{Var}(y^*)$ are functions of $x^*$ through $\boldsymbol{\phi}$, meaning that our prediction of $y^*$ after training is a gaussian with mean and variance which depend on $x^*$. Let's implement these results on the linear and non-linear datasets.

lamda = 1
X = np.array([[x_**d for d in range(0, 2)] for x_ in x_lin]) # X instantiated more elegantly here

prior_term = lamda*np.eye(2) # prior covariance matrix to include in MAP solution
w_maxlik = np.linalg.inv((X.T).dot(X) + prior_term).dot(X.T).dot(y_lin) # MAP weights to use in mean(y*)
var = np.sum((y_lin - X.dot(w_maxlik))**2)/x_lin.shape[0] # maximum-likelihood variance to use in var(y*)

S = np.linalg.inv((X.T).dot(X)/var + lamda*np.eye(2)) # posterior distribution covariance matrix
mu = S.dot(X.T).dot(y_lin)/var # posterior distribution mean vector

x_pred = np.linspace(-0.1, 1.1, 100)
X_pred = np.array([[x_**d for d in range(0, 2)] for x_ in x_pred])

mu_pred = X_pred.dot(mu) # calculate mean(y*)
stdev_pred = (np.sum(X_pred.dot(S)*X_pred, axis = 1) + var)**0.5 # calculate Var(y*)^0.5

plt.fill_between(x_pred, mu_pred + stdev_pred, mu_pred - stdev_pred, facecolor = 'grey', alpha = 0.5) # plot confidence intervals = +/- Var(y*)^0.5
plt.scatter(x_lin, y_lin, marker = 'x', color = 'red') # plot data
plt.plot(x_pred, mu_pred, color = 'black') # plot mean(y*)
beautify_plot({"title":"Bayesian regression predictive", "x":"$x$", "y": '$y$'})
plt.xlim([-0.1, 1.1])
plt.show()

The black line shows $\mathbb{E}[y^*]$, and the gray area shows $\pm \sqrt{\text{Var}(y^*)}$, i.e. plus/minus one standard deviation of the predictive distribution. The predictive uncertainty is a direct consequence in the uncertainty of $\mathbf{w}$ captured by the Bayesian approach. Uncertainty quantification is a great benefit of the bayesian approach because we can now justify our conclusions about the data in a principled way, unlike in the maximum-likelihood case where we only get an estimate for $y^*$ without a measure of how off this $y^*$ may be from the right answer. Let's apply this method to the non-linear dataset.

# exactly the same process with the linear case, except phi is different
lamda = 10**-5
D = 3
phi = np.array([[x_**d for d in range(0, D + 1)] for x_ in x_nonlin])

prior_term = lamda*np.eye(D + 1)
w_maxlik = np.linalg.inv((phi.T).dot(phi) + prior_term).dot(phi.T).dot(y_nonlin)
var = np.sum((y_nonlin - phi.dot(w_maxlik))**2)/x_nonlin.shape[0]

S = np.linalg.inv((phi.T).dot(phi)/var + lamda*np.eye(D + 1))
mu = S.dot(phi.T).dot(y_nonlin)/var

x_pred = np.linspace(-0.1, 1.1, 100)
phi_pred = np.array([[x_**d for d in range(0, D + 1)] for x_ in x_pred])

mu_pred = phi_pred.dot(mu)
stdev_pred = (np.sum(phi_pred.dot(S)*phi_pred, axis = 1) + var)**0.5

plt.fill_between(x_pred, mu_pred + stdev_pred, mu_pred - stdev_pred,
                 facecolor = 'grey', alpha = 0.5)
plt.scatter(x_nonlin, y_nonlin, marker = 'x', color = 'red')
plt.plot(x_pred, mu_pred, color = 'black')
beautify_plot({"title":"Bayesian regression predictive", "x":"$x$", "y": '$y$'})
plt.xlim([-0.1, 1.1])
plt.show()

## Update learning and visualizations

Until now we have been considering the whole dataset in one go. In this section we will be exploring how each point of in the dataset affects our knowledge about the model. This will expand our intuition about the bayesian approach and also show how we may go about *online learning* $-$ that is learning where the datapoints are gradually made available in a sequence. Online learning would for example be useful for scenarios such as weather prediction, where one can use each day's new data to improve the weather model.

Consider this scenario: we start off with a prior $p(\mathbf{w})$ and the data is made available on a point-by-point basis, in $(x_n, y_n)$ pairs. After observing one point, we can use Bayes' rule to evaluate the posterior, and use this to calculate the predictive:

$$
p(\mathbf{w}| y_1, x_1) \propto p(y_1| x_1, \mathbf{w}, \sigma^2)p(\mathbf{w})
$$

$$
p(y^* | x^*, y_1, x_1) = \int p(y^* | x^*, \mathbf{w}) p(\mathbf{w}|y_1, x_1) d\mathbf{w}
$$

When $(x_2, y_2)$ becomes available, we can use it to update our model. The weight distribution prior to seeing $(x_2, y_2)$ is $p(\mathbf{w}|y_1, x_1)$, reflecting the knowledge gained from $(x_1, y_1)$. Applying Bayes' rule again:

$$
p(\mathbf{w}| \{y_{n}, x_{n}\}^2_{n = 1}) \propto p(y_2| x_2, \mathbf{w}, \sigma^2)p(\mathbf{w}|y_1, x_1)
$$

$$
p(y^* | x^*, \{y_{n}, x_{n}\}^2_{n = 1}) = \int p(y^* | x^*, \mathbf{w}) p(\mathbf{w}|\{y_{n}, x_{n}\}^2_{n = 1}) d\mathbf{w}
$$

And in general after observing $N$ datapoints:

$$
p(\mathbf{w}| \{y_{n}, x_{n}\}^{N}_{n = 1}) \propto p(y_{N}| x_{N}, \mathbf{w}, \sigma^2)p(\mathbf{w}|y_{N-1}, x_{N-1})
$$

$$
p(y^* | x^*, \{y_{n}, x_{n}\}^{N}_{n = 1}) = \int p(y^* | x^*, \mathbf{w}) p(\mathbf{w}|\{y_{n}, x_{n}\}^N_{n = 1}) d\mathbf{w}
$$

**Note how the posterior of the $\mathbf{(N-1)^{th}}$ step becomes the prior for the $\mathbf{N^{th}}$ step.** This reflects the gradual increase in our knowledge about $\mathbf{w}$. It should be pointed out that for arbitrary distributions we must calculate the constant of proportionality for the posterior, or equivalently normalize $p(y^* | x^*, \{y_{n}, x_{n}\}^{N}_{n = 1})$ at the end to get a valid probability distribution $-$ this integral may be challenging and will often need to be approximated. Fortunately, in the special case of gaussian likelihoods with a conjugate prior we don't need to bother with this since the normalization constant is determined by the covariance matrix which we already know how to find.

The last outstanding point regarding this process is the noise level \\(\sigma\\). When learning the whole dataset at once, we used the \\(\sigma_{ML}\\) which maximised the likelihood as an estimate for the noise magnitude. It is possible to follow a fully Bayesian approach including a prior over \\(\sigma\\), but this is too involved for our purposes, so we will stick with the estimate \\(\sigma_{ML}\\) which maximises the likelihood of the observed points. For the first step where no points have yet been observed \\(\sigma_{ML}\\) would be \\(0\\) giving an infinitely sharp gaussian. To prevent this problem, which would otherwise constrain subsequent posteriors to be infinitely sharp as well, we will use an arbitrary noise level of \\(1\\).

Let's implement this method for the linear dataset since this has $2$ weights which we can easily visualise with a contour plot $-$ unlike the non-linear example which has several weights which we can't easily visualise in one go. At each step, will also draw $3$ weight samples from the posterior and plot the corresponding lines in data-space.

w1_range, w2_range = np.linspace(-5, 5, 100), np.linspace(-5, 5, 100) # specify the range of weights to be visualised
w1, w2 = np.meshgrid(w1_range, w2_range) # returns two 2d arrays with the values of w1_range and w2_range (see docs!)
grid = np.stack([w1, w2], axis = 2) # stack the two grids together to do aritmetic elegantly later

lamda = 4
no_points = 5 # do the algorithm for 3 sequential observations
prior = None # start with a null prior
noise = 1

colours = ['red', 'green', 'blue']
x_ = np.linspace(0, 1, 100)

for n in range(0, no_points + 1):

    X = np.array([[x_lin[n]**d for d in range(0, 2)]]) # X contains a single point (x[n]) for online learning
    
    if n == 0:
        prior_Q = np.sum(grid.dot(np.eye(2)/lamda)*grid, axis = 2) # quadratic form of prior (goes in the exponent)
        prior = np.exp(-1/2*prior_Q) # exponentiate quadratic form to get the prior
        
    lik_Q = np.sum((y_lin[n] - grid.dot(X.T))**2/0.5, axis = 2) # quadratic form of likelihood of (x[n], y[n]) point
    lik = np.exp(-1/2*lik_Q) # exponentiate quadratic form to get the likelihood of the new point
    
    post = prior*lik # posterior is prior*lik by Bayes' rule

    plt.figure(figsize = (12, 3)) # add new figure to plot this step
    for i, values in enumerate([prior, lik, post]):
        plt.subplot(1, 4, i + 1)
        plt.contourf(w1_range, w2_range, values, cmap = 'coolwarm', alpha = 0.5) # plot prior after n points
        remove_axes()
    
    for i in range(3):
        plt.subplot(143)
        w1, w2 = sample_weights_from(w2_range, w1_range, post)
        plt.scatter(w1, w2, marker = 'x', color = colours[i])
        
        plt.subplot(144)
        plt.plot(x_, w1 + w2*x_, color = colours[i])
        remove_axes()
    
    plt.subplot(144)
    plt.scatter(x_lin[:n+1], y_lin[:n+1], marker = 'o', facecolor = 'None', edgecolor = 'black')
    plt.ylim([0, 2])
    
    prior = post.copy() # set prior of the next step to be the posterior of the current step

    plt.show()

The first plot in the first row we see the prior before any data has been observed. After one datapoint is observed, our certainty about the values of the weights is improved: the likelihood (second plot, first row) multiplied by the prior gives a narrowed posterior (third plot, first row). Three independent weight samples are drawn (red, green and blue crosses) from the posterior giving the corresponding linear trendlines in data space (data as black crosses). In the next step **the prior is the posterior of the previous step**, i.e. $3^{rd}$ plot of $n^{th}$ row is the same as the $1^{st}$ plot of the $(n+1)^{th}$ row. Note that as more datapoints are added the posterior narrows down, and the weights are constrained to a progressively narrower area. This is reflected in the data space where the red/green/blue lines are also progressively constrained.