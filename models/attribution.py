import tensorflow as tf
from src.utils import tf_entropy
from src.utils import Bunch


def logistic_attribution(X, w, b):
  '''
  :param X: input matrix, shape (?nB, nF)
  :param w: weights, shape (nF, 1)
  :param b: bias, scalar
  :return:
  '''

  w = tf.reshape(w, [-1,1]) # force to be a column (nF, 1)

  Xi_dot_w = tf.matmul(X, w)  # (nB, 1)
  logits = Xi_dot_w + b       # (nB, 1)
  prob_diff = tf.sigmoid(logits) - tf.sigmoid(b)  # (nB, 1)
  Xij_wj = X * tf.squeeze(w) # broadcast + elementwise mult -> (nB, nF)
  logit_fracs = Xij_wj / Xi_dot_w # (nB, nF)
  attribs = tf.abs(prob_diff * logit_fracs) # (nB, nF)
  return attribs

def label_corr_stats(X, y):
  '''
   Stats for streaming compute of corr of (exploded) features in X with
   labels y
  :param X: (?nB, nF)
  :param y: (?nB)
  :return: (nF)
  '''
  Xy_av = tf.reduce_mean(X * y, axis=0)
  X_av = tf.reduce_mean(X, axis=0)
  y_av = tf.reduce_mean(y, axis=0)
  Xsq_av = tf.reduce_mean(X * X, axis=0)
  ysq_av = tf.reduce_mean(y * y, axis=0)
  return Bunch(xy=Xy_av, x=X_av, y=y_av, xsq=Xsq_av, ysq=ysq_av)



def logistic_afvi(X, w, b):
  w = tf.squeeze(w) # force to be a column (nF, 1)
  Xij_wj = X * w  # broadcast + elementwise mult -> (nB, nF)
  afvi =  tf.abs(tf.sigmoid(Xij_wj + b) - tf.sigmoid(b))
  return tf.reduce_mean(afvi, axis=0)
