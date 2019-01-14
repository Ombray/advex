import numpy as np
from abc import ABC, abstractmethod
from src.utils import randnsphere, Bunch
import tensorflow as tf

class DataGenerator(ABC):
  def __init__(self, config: Bunch = None):
    '''
    This needs to ensure the following attributes are available:
    X = float32 tensor N rows x m features
    Y = float32 N x 1
    config must have at least the following:
    config.N = number of rows
    config.batch_size = mini-batch size for training

    :param config: any needed configs
    '''
    self.config = config
    super().__init__()

  # Pack feature-dict into single tensor
  # Only needed if we're getting
  # columnns as a dict out of a dataframe
  @staticmethod
  def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    f_vals = map(lambda x: tf.cast(x, tf.float32), features.values())
    labels = tf.cast(labels, tf.float32)
    features = tf.stack(list(f_vals), axis=1)
    return features, tf.reshape(labels, [-1,1])

  def data(self, l2_epsilon=0.0):
    d = self.X.shape[1]
    arr = self.X + randnsphere(self.config.N, d) * l2_epsilon
    ds = tf.data.Dataset.from_tensor_slices((arr, self.Y))
    ds = ds.shuffle(10).batch(self.config.batch_size)
    return ds

  def next_batch(self, l2_epsilon=0.0):
    inputs, labels = next(iter(self.data(l2_epsilon)))
    inputs = tf.cast(inputs, tf.float32)
    labels = tf.reshape(labels, [-1, 1])
    return inputs, labels


