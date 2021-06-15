def get_plot_title_bayesian_online_learning(n, i):
    
    if i == 0 and n == 0:
        return "$p(\mathbf{w})$"

    if i == 1 and n == 0:
        return "$p(y_1|x_1,\mathbf{w})$"

    if i == 2 and n == 0:
        return "$p(\mathbf{w}|y_1, x_1)$"

    if i == 0 and n == 1:
        return "$p(\mathbf{w}|y_1, x_1)$"

    if i == 1 and n == 1:
        return "$p(y_2 | x_2, \mathbf{w})$"

    if i == 2 and n == 1:
        return "$p(\mathbf{w} | y_{1:2}, x_{1:2})$"

    if i == 0 and n > 1:
        return "$p(\mathbf{w}|y_{1:" + str(n) + "}, x_{1:" + str(n) + "})$"

    if i == 1 and n > 1:
        return "$p(y_{" + str(n) + "} | x_{" + str(n) + "}, \mathbf{w})$"

    if i == 2 and n >1 :
        return "$p(\mathbf{w} | y_{1:" + str(n) + "}, x_{1:" + str(n) + "})$"