import tensorflow as tf
from src.utils import tf_entropy, num_above_relative_threshold
from src import utils
from src.robust_logistic import RobustLogisticModel
from models import attribution

def binary_classification_model(features, labels, mode, params: utils.Bunch):
  """Custom model; initially just linear (logistic or poisson)
  """
  params = utils.Bunch(**params)
  optimizer = params.get('optimizer', 'ftrl')
  l1_reg = params.get('l1_reg', 0.0)
  l2_reg = params.get('l2_reg', 0.0)
  tf.set_random_seed(123)
  labels = tf.cast(labels, tf.float32)
  epsilon = params.perturb_norm_bound
  norm_order = params.get('perturb_norm_order', 2)
  train_perturb_frac = params.train_perturb_frac
  test_perturb_frac = params.test_perturb_frac
  # do the various feature transforms according to the
  # 'feature_column' param, so now we have the feature-vector
  # that we will do computations on.
  x = tf.feature_column.input_layer(features, params.feature_columns)
  # for units in params['hidden_units']:
  #   net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

  # Compute logits (1 per class).
  #logits = tf.layers.dense(net, params['n_classes'], activation=None)
  #logits = tf.layers.dense(net, 1, activation=None, name='dense')
  dense = tf.layers.Dense(1, activation=None,
                          kernel_initializer=\
                            tf.keras.initializers.zeros(),
                            #tf.keras.initializers.RandomNormal(seed=123),
                          bias_initializer= \
                            tf.keras.initializers.zeros())
                            #tf.keras.initializers.RandomNormal(seed=123))

  if len(dense.trainable_variables) == 0:
    dense(x) # to force the kernel initialization
  # this is the "kernel" i.e. weights, does not include bias
  coefs = dense.trainable_variables[0]
  bias = dense.trainable_variables[1][0]
  perturb_frac = train_perturb_frac if mode == tf.estimator.ModeKeys.TRAIN \
    else test_perturb_frac
  x_perturbed, _  = RobustLogisticModel.perturb_continuous(
    x, labels, coefs,
    norm_bound=epsilon,
    norm_order=norm_order,
    perturb_frac=perturb_frac,
    seed=123)
  logits = dense(x_perturbed)
  if params.activation == 'sigmoid':
    predictions = tf.sigmoid(logits)
  elif params.activation == 'sign':
    predictions = tf.maximum(0.0, tf.sign(logits))
  else: # assume relu
    predictions = tf.nn.relu(logits)
  labels = tf.reshape(labels, [-1,1])
  # Compute predictions.
  predicted_classes = tf.maximum(tf.sign(predictions - 0.5), 0)

  # if mode == tf.estimator.ModeKeys.PREDICT:
  #   predictions = {
  #     'class_ids': predicted_classes[:, tf.newaxis],
  #     'probabilities': tf.nn.softmax(logits), # not really used
  #     'logits': logits,
  #   }
  #   return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Compute loss.
  if params.activation == 'sigmoid':
    loss = tf.reduce_mean(
      tf.keras.backend.binary_crossentropy(target=labels,
                                           output=logits,
                                           from_logits=True))
  elif params.activation == 'sign':
    loss = tf.reduce_mean(- (2*labels - 1) * logits )
  else:
    raise Exception(f'loss not known for activation {params.activation}')

  if l1_reg > 0 and optimizer != 'ftrl':
    loss = loss + l1_reg * tf.norm(coefs, ord=1)
  if l2_reg > 0 and optimizer != 'ftrl':
    loss = loss + l2_reg * tf.sqrt(tf.maximum(0.0, tf.nn.l2_loss(coefs)))

  adv_reg_lambda = params.get('adv_reg_lambda', 0.0)
  if adv_reg_lambda and perturb_frac > 0.0:
    clean_logits = dense(x)
    clean_loss = tf.reduce_mean(
      tf.keras.backend.binary_crossentropy(target=labels,
                                           output=clean_logits,
                                           from_logits=True))
    loss = clean_loss + adv_reg_lambda * loss

# Compute evaluation metrics.
  accuracy = tf.metrics.accuracy(labels=labels,
                                 predictions=predicted_classes,
                                 name='acc_op')

  auc = tf.metrics.auc(labels=labels, predictions=predictions, name = 'auc-op')



  # add metrics etc for tensorboard
  tf.summary.scalar('accuracy', accuracy[1])
  tf.summary.scalar('auc', auc[1])
  tf.summary.scalar('loss', loss)

  if mode == tf.estimator.ModeKeys.EVAL:
    # axiomatic attribution (Integrated Grads)
    feat_val_attribs = attribution.logistic_attribution(x, coefs, bias)
    feat_val_corr_stats = attribution.label_corr_stats(x, labels)
    av_attribs = tf.reduce_mean(feat_val_attribs, axis=0)
    attrib_entropy = tf_entropy(feat_val_attribs)
    num_high_attribs = num_above_relative_threshold(
      feat_val_attribs, thresh=params.get('thresh', 0.1))
    av_attrib_entropy = tf.metrics.mean(attrib_entropy)
    av_high_attribs = tf.metrics.mean(num_high_attribs)
    mean_attribs = tf.metrics.mean_tensor(av_attribs, name='attrib')
    xy_av = tf.metrics.mean_tensor(feat_val_corr_stats.xy, name='xy_av')
    x_av = tf.metrics.mean_tensor(feat_val_corr_stats.x, name='x_av')
    y_av = tf.metrics.mean_tensor(feat_val_corr_stats.y, name='y_av')
    xsq_av = tf.metrics.mean_tensor(feat_val_corr_stats.xsq, name='xsq_av')
    ysq_av = tf.metrics.mean_tensor(feat_val_corr_stats.ysq, name='ysq_av')

    # ad-hoc attribution (AFVI)
    afvi = attribution.logistic_afvi(x, coefs, bias)
    mean_afvi = tf.metrics.mean_tensor(afvi, name='afvi')

    metrics = dict(accuracy=accuracy,
                   auc=auc,
                   attrib_ent=av_attrib_entropy,
                   high_attribs=av_high_attribs,
                   attrib=mean_attribs,
                   afvi=mean_afvi,
                   xy_av=xy_av,
                   x_av=x_av,
                   y_av=y_av,
                   xsq_av=xsq_av,
                   ysq_av=ysq_av)

    # the histograms don't work in eval mode??
    tf.summary.histogram('attrib', mean_attribs[1])
    tf.summary.histogram('afvi', mean_afvi[1])

    return tf.estimator.EstimatorSpec(
      mode, loss=loss, eval_metric_ops=metrics)

  # Create training op.
  assert mode == tf.estimator.ModeKeys.TRAIN

  #
  if optimizer == 'adam':
    loss_optimizer = tf.train.AdamOptimizer(learning_rate=params.lr)
  elif optimizer == 'ftrl':
    loss_optimizer = tf.train.FtrlOptimizer(learning_rate=params.lr,
                                    l1_regularization_strength=l1_reg,
                                    l2_regularization_strength=l2_reg)
  elif optimizer == 'adagrad':
    loss_optimizer = tf.train.AdagradOptimizer(learning_rate=params.lr)
  elif optimizer == 'sgd':
    loss_optimizer = tf.train.GradientDescentOptimizer(
       learning_rate=params.lr)
  else:
    raise Exception(f"Unknown optimizer: {optimizer}")

  train_op = loss_optimizer.minimize(loss,
                                   global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode, loss=loss,
                                    train_op=train_op)
