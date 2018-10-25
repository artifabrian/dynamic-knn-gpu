import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', size=20)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('legend', fontsize=20)
matplotlib.rc('figure', titlesize=20)


def plot_data(X, y):
    """
        Plots the data found in X with the y classes.
        Expects exactly two classes.

        :type X: np.array
        :type y: np.array
        :rtype: None
    """
    y = y.flatten()

    x_point_class_0 = X[y == 0].T[0].T
    y_point_class_0 = X[y == 0].T[1].T

    x_point_class_1 = X[y == 1].T[0].T
    y_point_class_1 = X[y == 1].T[1].T

    plt.plot(x_point_class_0, y_point_class_0, 'ro', label='Class 0')
    plt.plot(x_point_class_1, y_point_class_1, 'go', label='Class 1')
    plt.legend()
    plt.show()


def plot_data_with_test_point(X, y, test_point):
    """
        Plots the data found in X with the y classes.
        Also plots the test points.
        Expects exactly two classes.

        :type X: np.array
        :type y: np.array
        :type test_point: np.array
        :rtype: None
    """

    y = y.flatten()

    x_point_class_0 = X[y == 0].T[0].T
    y_point_class_0 = X[y == 0].T[1].T

    x_point_class_1 = X[y == 1].T[0].T
    y_point_class_1 = X[y == 1].T[1].T

    x_point_test = test_point.T[0].T
    y_point_test = test_point.T[1].T

    plt.plot(x_point_class_0, y_point_class_0, 'ro', label='Class 0')
    plt.plot(x_point_class_1, y_point_class_1, 'go', label='Class 1')
    plt.plot(x_point_test, y_point_test, 'bo', label='Test Point')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_data(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
    plot_data_with_test_point(np.array([[1, 2], [3, 4]]), np.array([0, 1]), np.array([[2, 3]]))