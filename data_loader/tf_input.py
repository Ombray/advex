import tensorflow as tf
import numpy as np

def input_fn(X, y, col_spec,
             num_epochs=10, shuffle=False,
             batch_size=20):
  """Generate an input function for the Estimator."""
  tf.set_random_seed(123)
  np.random.seed(123)

  nrows = X.shape[0]
  column_names = [s['name'] for s in col_spec]
  X = list(zip(column_names, X.transpose()))
  for i,s in enumerate(col_spec):
    if s['type'] == 'cat':
      X[i] = (s['name'], X[i][1].astype(np.int64))
  # Extract lines from input files using the Dataset API.
  X = dict(X)
  dataset = tf.data.Dataset.from_tensor_slices((X, y))
  if shuffle:
    dataset = dataset.shuffle(buffer_size=nrows, seed=123)
  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset
