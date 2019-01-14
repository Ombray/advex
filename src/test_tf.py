import tensorflow as tf
# test tf.gather

indices = tf.constant([[0], [2], [1]])

# params: one set of per column
params = tf.constant([10,11,12,13])

tf.gather(params, indices)