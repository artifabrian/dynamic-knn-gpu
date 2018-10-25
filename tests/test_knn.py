import random
import unittest
import tensorflow as tf
import time

from metrics.distances import euclidean_distance
from model.knn import build_knn_graph
from utils.generator import generate_pseudo_random_data


class TestKNN(unittest.TestCase):

    def test_euclidean_distance(self):
        session = tf.Session()
        session.run(tf.global_variables_initializer())

        a = tf.constant([[1,2,3],[4,5,6]], dtype=tf.float32)
        b = tf.constant([1, 2, 3], dtype=tf.float32)
        distance_graph = euclidean_distance(a, b)

        distance = session.run(distance_graph)

        session.close()

        print("Distance: %s" % distance)

        self.assertEqual(distance[0], 0)
        self.assertAlmostEqual(distance[1], 5.196, 2)

    def test_model(self):
        # Generate some data.
        X, y = generate_pseudo_random_data(number_of_training_points=200, n_features=2)

        # Build the graph.
        class_pred, counts, X_train, y_true, X_test, k = build_knn_graph(X.shape[0], X.shape[1], gpu_enable=True)

        # Example test point
        test_point_1 = [100.0, 134.0]
        test_point_2 = [-50.0, 54.0]
        test_point_3 = [-150.0, -50.0]
        test_point_4 = [150.0, 254.0]

        # Run a session.
        # (log_device_placement=True => verbose output of where each operation is mapped)
        # (gpu_options.allow_growth = True => attempts to allocate only as much GPU memory based on runtime allocations)
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True

        session = tf.Session(config=config)
        session.run(tf.global_variables_initializer())

        test_point_1_pred = session.run(class_pred, feed_dict={X_train: X, y_true: y, X_test: test_point_1, k: 15})
        test_point_2_pred = session.run(class_pred, feed_dict={X_train: X, y_true: y, X_test: test_point_2, k: 15})
        test_point_3_pred = session.run(class_pred, feed_dict={X_train: X, y_true: y, X_test: test_point_3, k: 15})
        test_point_4_pred = session.run(class_pred, feed_dict={X_train: X, y_true: y, X_test: test_point_4, k: 15})

        session.close()

        # Show results!
        print("Predicted class for test point 1: %s" % test_point_1_pred)
        print("Predicted class for test point 2: %s" % test_point_2_pred)
        print("Predicted class for test point 2: %s" % test_point_3_pred)
        print("Predicted class for test point 2: %s" % test_point_4_pred)

        self.assertEqual(test_point_1_pred, 0)
        self.assertEqual(test_point_2_pred, 1)
        self.assertEqual(test_point_1_pred, 1)
        self.assertEqual(test_point_2_pred, 0)



if __name__ == '__main__':
    unittest.main()