import numpy as np
import tensorflow as tf
import metrics.distances as distances
from utils.generator import generate_pseudo_random_data
from utils.plots import plot_data_with_test_point


def build_knn_graph(number_of_training_points, number_of_features,
                    distance_metric=distances.euclidean_distance,
                    gpu_enable=False, include_prints=False):
    """
        Classifier implementing the k-nearest neighbors vote.

        Returns a TensorFlow graph, that has the training data,
        true y values, a single test point to predict and the k value
        as placeholders.

        Supports a single CPU or GPU.

        The distance metric should return a sub-graph calculating the distance
        between an array of values and a single value.

        :type number_of_training_points: int
        :type number_of_features: int
        :type distance_metric: function
        :type gpu_enable: bool
        :type include_prints: bool

        :rtype class_pred: tf.Tensor
        :rtype y_counts: tf.Tensor
        :rtype X_train: tf.Placeholder
        :rtype y_true: tf.Placeholder
        :rtype X_test: tf.Placeholder
        :rtype k: tf.Placeholder
    """

    device = '/cpu:0'
    if gpu_enable:
        device = '/gpu:0'

    with tf.device(device):

        X_train = tf.placeholder(tf.float32, shape=[number_of_training_points, number_of_features],
                                 name="X_train_placeholder")

        y_true = tf.placeholder(tf.float32, shape=[number_of_training_points, 1],
                                name="y_true_placeholder")

        # Expects a single instance.
        X_test = tf.placeholder(tf.float32, shape=[number_of_features], name="X_test_placeholder")

        # k is a scalar placeholder
        k = tf.placeholder(tf.int32, shape=(), name="k")

        # Calculates the distance from the test instance against the training data.
        distances = distance_metric(X_train, X_test)

        # Negating the distances.
        # (We need this trick because TensorFlow has top_k API and no closest_k or reverse=True api.)
        neg_distances = tf.negative(distances)

        # It returns the 'k' values and indexes from the least distant nodes.
        values, indexes = tf.nn.top_k(neg_distances, k)

        if not gpu_enable and include_prints:
            indexes = tf.Print(indexes, [values, indexes], message="Top k Results:")

        # We gather the classes using the indexes in the y_true tensor.
        # (It works as an efficient "multiple get".)
        y_neighbours = tf.gather(y_true, indexes)

        # Since the class is discrete we cast it into an int32.
        y_neighbours = tf.cast(y_neighbours, tf.int32)

        # We aggregate sums of the values and calculate how many values from each class is there on the neighbourhood.
        # (i.e. creates a tensor with the count on each index for each class)
        y_counts = tf.bincount(y_neighbours)

        if not gpu_enable and include_prints:
            y_counts = tf.Print(y_counts, [y_counts, y_neighbours], message="Counts:")

        # Gets the index (i.e. corresponding class) with the max count.
        # (Unfortunately this operation doesn't seem to work on GPU as of TensorFlow 1.11.0
        #  An alternative to this is to use "allow_soft_placement=True")
        with tf.device('/cpu:0'):
            class_pred = tf.argmax(y_counts)

    return class_pred, y_counts, X_train, y_true, X_test, k

if __name__ == "__main__":
    # Generate some data.
    X, y = generate_pseudo_random_data(number_of_training_points=200, n_features=2)

    # Build the graph.
    class_pred, counts, X_train, y_true, X_test, k = build_knn_graph(X.shape[0], X.shape[1], gpu_enable=True)

    # Example test point
    test_point_1 = [100.0, 134.0]
    test_point_2 = [-50.0, 54.0]

    # Run a session.
    # (log_device_placement=True => verbose output of where each operation is mapped)
    # (gpu_options.allow_growth = True => attempts to allocate only as much GPU memory based on runtime allocations)
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True

    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())

    test_point_1_pred = session.run(class_pred, feed_dict={X_train: X, y_true: y, X_test: test_point_1, k: 15})
    test_point_2_pred = session.run(class_pred, feed_dict={X_train: X, y_true: y, X_test: test_point_2, k: 15})

    session.close()

    # Show results!
    print("Predicted class for test point 1: %s" % test_point_1_pred)
    print("Predicted class for test point 2: %s" % test_point_2_pred)
    plot_data_with_test_point(X, y, np.array([test_point_1, test_point_2]))