from model.knn import build_knn_graph
from utils.generator import generate_pseudo_random_data
import tensorflow as tf
import numpy as np
from utils.plots import plot_data_with_test_point

# Generate some data.
X, y = generate_pseudo_random_data(number_of_training_points=200, n_features=2)

# Build the k-nn graph.
class_pred, counts, X_train, y_true, X_test, k = build_knn_graph(X.shape[0], X.shape[1], gpu_enable=True)

# Example test points
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