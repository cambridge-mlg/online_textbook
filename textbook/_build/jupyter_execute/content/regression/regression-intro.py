# Regression 

Regression is a class of supervised machine learning tasks in which the aim is to predict a **real valued output** $y^\star$, given an input vector $\mathbf{x}^\star$ using a **training set of input-output pairs** $\{\mathbf{x}_n, y_n\}_{n=1}^N$. Each vector $\mathbf{x}_n$ represents a point in some $D$-dimensional space, where $D$ is the number of input parameters being measured.

````{margin}
Regression Jargon: The inputs $\mathbf{x}_n$ are also known as the features, covariates, or independent variables.
The outputs $y_n$ are also known as the responses, targets, or dependent variables. 
````

[//]: # (The line above might be better as a margin comment)

Here's a cartoon regression task for a one-dimensional dataset, i.e. each $\mathbf{x}_n$ is a one-dimensional vector, or just a scalar value. Often the full solution to a regression task involves returning to the user a best guess for $y^\star$ along with a measure of uncertainty, the latter being important for decision making. 

![](./imgs/intro-regression.png)

Regression encompasses many different types of input e.g. they can be scalar or multi-dimensional; real valued or discrete. Regression problems lie on a sliding scale from those that involve **interpolation** (the required predictions are typically close to the training data) to **extrapolation** (the required predictions are typically far from the training data). Example regression problems include:

|Application|Inputs|Outputs| 
|:-:|:-:|:-:|
|computer aided drug design|representation of molecule| biological activity of molecule|  
|solar power supply forecasting| time, physical models, satellite images | solar power supply|  
|informing health care policy| health care center characteristics|patient treatment outcomes|
|image super resolution|low resolution image|high resolution image|
|system identification|system's state at time \\(t-1\\)|system's state at time \\(t\\)|


The last two examples above are **multi-output regression problems** which involve predicting a vector of real valued outputs, rather than just a single scalar output.

In regression, the predictions themselves may not always be the central focus. For example, in the 'informing health care policy' example, the goal is to assess the contribution from different input variables (health centre funding level, availability of treatments, population demographic etc.) on the treatment outcome. 

Regression also serves as a good introduction to several cross-cutting concepts in inference and machine learning including *generative models*, *maximum likelihood estimation*, *overfitting* and *probabilistic inference*. 

## Outline of subsequent sections

1. [Linear regression](regression-linear.ipynb): Linear regression, least squares and maximum likelihood fitting
2. [Non-linear regression](regression_non_linear.ipynb): Non-linear regression using basis functions
3. [Overfitting in non-linear regression](regression_overfitting.ipynb): What is overfitting, diagnosis using validation sets
4. [Regularised non-linear regression](regression_regularisation.ipynb): Using regularisation to mitigate overfitting, interpretations of regularisation as MAP inference
5. [Bayesian non-linear regression](regression_bayesian.ipynb): Bayesian approaches to regression that return uncertainty in the parameter estimates
6. [Visualising Bayesian linear regression: Online learning](regression_bayesian-online-visualisations.ipynb): Visualising how the posterior distribution evolves as data arrive




```{toctree}
:hidden:
:titlesonly:


regression-linear
regression-nonlinear
Overfitting <regression-overfitting>
Regularisation <regression-regularisation>
Bayesian Regression <regression-bayesian>
```
