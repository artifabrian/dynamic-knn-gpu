import numpy as np

from utils.plots import plot_data


def generate_pseudo_random_data(number_of_training_points=350, n_features=2,
                                class_0_mean=150, class_0_std=32,
                                class_1_mean=-15, class_1_std=27):
    """
        Generates circular-shaped data for testing.
        Works for two classes.

        :type number_of_training_points: int
        :type n_features: int
        :type class_0_mean: int
        :type class_0_std: int
        :type class_1_mean: int
        :type class_1_std: int
        :rtype X: np.array
        :rtype y: np.array
    """

    number_of_training_points_per_class = int(number_of_training_points / 2)

    X = []
    y = []

    # TODO: make this more efficient.
    for i in range(number_of_training_points_per_class):
        instance_class_zero = []

        for j in range(n_features):
            val = np.random.normal(class_0_mean, class_0_std)
            instance_class_zero.append(val)

        instance_class_one = []
        for j in range(n_features):
            val = np.random.normal(class_1_mean, class_1_std)
            instance_class_one.append(val)

        X.append(instance_class_zero)
        y.append([0])  # Class 0 label.

        X.append(instance_class_one)
        y.append([1])  # Class 1 label.

    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = generate_pseudo_random_data(number_of_training_points=500, n_features=3)
    plot_data(X, y)