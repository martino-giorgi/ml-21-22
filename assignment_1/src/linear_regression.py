import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def function(x, theta):
    y = theta[0] + theta[1] * x[0] + theta[2] * x[1] + theta[3] * np.cos(x[1]) + theta[4] * pow(x[0], 2)
    return y


def lmse(y, y_est):
    sum_of_squares = 0
    for i in range(len(y)):
        sum_of_squares += (y[i] - y_est[i]) ** 2

    return sum_of_squares / len(y)


def main():
    # Load data
    data = np.load('../data/data.npz')
    x = data['x']
    y = data['y']

    ax = plt.axes(projection='3d')

    ax.scatter3D(x[:, 0], x[:, 1], y, color="green", alpha=0.2)

    ones_col_vec = np.ones(shape=(y.shape[0], 1))

    # Create column vector of cos(x[1]) with shape as (2000,1)
    cos_col_vec = np.cos(x[:, 1]).reshape(2000, 1)

    # Create column vector of x[0]^2 with shape as (2000,1)
    x_square_col_vec = np.power(x[:, 0], 2).reshape(2000, 1)

    # Form X matrix with shape as (2000,5)
    X = np.hstack((ones_col_vec, x, cos_col_vec, x_square_col_vec))

    lr = LinearRegression(fit_intercept=False)

    # Estimate parameters
    lr.fit(X, y)
    theta_hat = lr.coef_

    # Estimate y
    y_est = []
    for x_couple in x:
        y_est.append(function(x_couple, theta_hat))

    # Calculate error by LME
    print("LME: ", lmse(y, y_est))

    # Plot estimated y on the same graph
    ax.scatter3D(x[:, 0], x[:, 1], y_est, color="red", alpha=0.2)
    plt.show()


if __name__ == '__main__':
    main()
