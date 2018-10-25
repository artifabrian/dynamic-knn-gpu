import tensorflow as tf

def example_distance(a, b):
    """
        Generates a graph for an example distance metric.

        The argument 'a' should be a list of Tensors and 'b' a single tensor.

        :type a: list
        :type b: tf.Tensor
        :rtype : tf.Tensor
    """

    return tf.reduce_sum(tf.abs(tf.subtract(a, b)), axis=1)

def euclidean_distance(a, b):
    """
        Generates a graph for the euclidean distance.

        The argument 'a' should be a list of Tensors and 'b' a single tensor.

        :type a: list
        :type b: tf.Tensor
        :rtype : tf.Tensor
    """

    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(a, b)), axis=1))
