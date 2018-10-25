import random
import timeit
import unittest
import tensorflow as tf
import time
from model.knn import build_knn_graph
from utils.generator import generate_pseudo_random_data

config = tf.ConfigProto()

def test_tf(X, y, test_points, k_values, gpu_enable=True, print_results=True):
    start = time.time()

    # Run a session.
    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())

    # Build the graph.
    class_pred, counts, X_train, y_true, X_test, k = build_knn_graph(X.shape[0], X.shape[1], gpu_enable=gpu_enable)

    for data_point in zip(test_points, k_values):

        start_run = time.time()
        test_point_pred = session.run(class_pred,
                                      feed_dict={X_train: X, y_true: y, X_test: data_point[0], k: data_point[1]})
        end_run = time.time()

        # Show results!
        if print_results:
            print("Predicted class for test point %s(...) using k=%s: %s" %
                  (data_point[0][:10], data_point[1], test_point_pred))

            print("Run Time elapsed: %f" % (end_run - start_run))

    session.close()

    end = time.time()
    print("Time elapsed: %f" % (end - start))

    return end - start


class TestKNNPerformanceTensorFlowCPUvsGPU(unittest.TestCase):

    def test_small_dataset(self):
        # Data params
        number_of_training_points = 10000
        n_features = 1000
        n_test_points = 100

        # Example test point
        test_points = []
        for i in range(n_test_points):
            test_points.append(random.choices(list(range(-100, 200)), k=n_features))

        # Dynamic k
        k_values = random.choices(list(range(min([number_of_training_points, 10000]))),
                                       k=n_test_points)

        X, y = generate_pseudo_random_data(number_of_training_points=number_of_training_points, n_features=n_features)

        elapsed_time_cpu = test_tf(X, y, test_points, k_values, gpu_enable=False, print_results=False)
        elapsed_time_gpu = test_tf(X, y, test_points, k_values, gpu_enable=True, print_results=False)

        print('TF CPU (s):')
        print(elapsed_time_cpu)
        print('TF GPU (s):')
        print(elapsed_time_gpu)
        print('TF GPU speedup over TF CPU: {}x'.format(elapsed_time_cpu / elapsed_time_gpu))

        self.assertGreater(elapsed_time_cpu, elapsed_time_gpu)

    def test_larger_dataset(self):
        # Data params
        number_of_training_points = 100000
        n_features = 100
        n_test_points = 1000

        # Example test point
        test_points = []
        for i in range(n_test_points):
            test_points.append(random.choices(list(range(-100, 200)), k=n_features))

        # Dynamic k
        k_values = random.choices(list(range(min([number_of_training_points, 10000]))),
                                  k=n_test_points)

        X, y = generate_pseudo_random_data(number_of_training_points=number_of_training_points, n_features=n_features)

        elapsed_time_cpu = test_tf(X, y, test_points, k_values, gpu_enable=False, print_results=False)
        elapsed_time_gpu = test_tf(X, y, test_points, k_values, gpu_enable=True, print_results=False)

        print('TF CPU (s):')
        print(elapsed_time_cpu)
        print('TF GPU (s):')
        print(elapsed_time_gpu)
        print('TF GPU speedup over TF CPU: {}x'.format(elapsed_time_cpu / elapsed_time_gpu))

        self.assertGreater(elapsed_time_cpu, elapsed_time_gpu)

if __name__ == '__main__':
    unittest.main()