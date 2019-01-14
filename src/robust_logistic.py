import tensorflow as tf
import numpy as np
from src.utils import tf_numpy
import math



class RobustLogisticModel(tf.keras.layers.Layer):
  def __init__(self, num_outputs, config):
    '''
    :param num_outputs:
    :param l2_epsilon: bound on l2_norm of perturbation delta
    :param column_spec: list of dicts, e.g.
       [ dict(type='cat', card=3), dict(type='num', min=0, max=4), ... ]
    '''
    self.config = config
    self.l2_epsilon = config.get('l2_epsilon', 0.0)

    # Note if num_perturbed_cateogoricals is explicitly given, then this
    #  is applied separately
    # to the categoricals (when perturb_one_hot is FALSE), and the numerical
    # perturbs are subject to l2_epsilon.
    # When perturb_one_hot is TRUE, then l2_epsilon is applied to the
    # one-hot transformed input as a whole

    # max categoricals to perturb
    self.num_perturbed_categoricals = config.get('num_perturbed_categoricals')


    self.seed = config.get('seed', 123)
    super(RobustLogisticModel, self).__init__()
    # column specs for all columns except label
    column_spec = [spec for spec in config.col_spec if
                   spec['name'] != config.get('label_name', None)]
    self.column_spec = column_spec
    self.n_input_features = len(column_spec)
    self.cat_columns = [k for k in range(len(column_spec)) if
                        column_spec[k]['type'] == 'cat']
    self.numeric_columns = [k for k in range(len(column_spec)) if
                            column_spec[k]['type'] != 'cat']
    self.cat_mask = np.zeros([self.n_input_features])
    self.numeric_mask = np.zeros([self.n_input_features])
    self.cat_mask[self.cat_columns] = 1
    self.numeric_mask[self.numeric_columns] = 1
    # total num inputs after 1-hot
    self.num_inputs = sum([spec['card'] if spec['type'] == 'cat' else 1
                           for spec in column_spec])
    self.num_outputs = num_outputs
    # indices of coefs
    self.coef_index_starts = []
    self.coef_index_ends = []
    curr = 0
    for i in range(len(column_spec)):
      self.coef_index_starts += [curr]
      if column_spec[i]['type'] == 'cat':
        card = column_spec[i]['card']
        self.coef_index_ends += [curr + card - 1]
        curr += card
      else:
        self.coef_index_ends += [curr]
        curr += 1
    self.numeric_coef_indices = \
      np.array(self.coef_index_starts)[self.numeric_columns]
    self.lower_bounds = [column_spec[i].get('min', -1e6)
                         for i in self.numeric_columns]
    self.upper_bounds = [column_spec[i].get('max',  1e6)
                         for i in self.numeric_columns]


  def build(self, input_shape):
    self.kernel = self.add_variable("kernel",
                                    shape=[self.num_inputs,
                                           self.num_outputs],
                                    dtype=tf.float32,
                                    initializer=tf.keras.initializers.RandomNormal(seed=self.seed))


    self.bias = self.add_variable("bias",
                                  shape=[self.num_outputs],
                                  dtype=tf.float32,
                                  initializer=tf.keras.initializers.RandomNormal(seed=self.seed))

  def coefs_by_feature(self):
    '''
    get coefs grouped by (pre-1-hot) input feature as a list,
    where the i'th entry is:
    - an ordered list of coefficients if i'th feature is categorical
      (the j'th value in the list is the coef for index j of cat feature i)
    - a list of length 1 if the feature is numerical
    :return: list as described above
    '''
    grouped_coefs = []
    coefs = tf.squeeze(tf.reshape(self.kernel, [1,-1])).numpy()
    for col in range(self.n_input_features):
      # start, end indices of transformed feature
      start, end = self.coef_index_starts[col], self.coef_index_ends[col]
      grouped_coefs += [coefs[start:(end + 1)]]
    return grouped_coefs

  def set(self, coefs, bias=0.0):
    '''
    Set coefs, bias to given values
    :param coefs:
    :param bias:
    :return:
    '''
    tf.keras.backend.set_value(self.kernel, np.array(coefs).reshape([-1,1]))
    tf.keras.backend.set_value(self.bias, np.array([bias]))


  def get(self):
    '''
    Return weights
    :return:
    '''
    coefs = self.kernel.numpy().squeeze()
    bias = self.bias.numpy()
    return dict(coefs = coefs, bias = bias)

  def one_hot(self, input, col_index):
    orig_column = input[:, col_index]
    cardinality = self.column_spec[col_index]['card']
    return tf.one_hot(tf.cast(orig_column, tf.int32), cardinality)

  def one_hot_all(self, input):
    transformed_cols = []
    for i in range(input.shape[-1]):
      if i in self.cat_columns:
        col_i = self.one_hot(input, i)
      else:
        col_i = tf.reshape(input[:, i], [-1,1])
      col_i = tf.cast(col_i, tf.float32)
      transformed_cols += [ col_i ]
    return tf.concat(transformed_cols, axis=1)


  def perturb_numeric(self, inputs, labels, robust=1.0, indicators=None):
    '''
    Adversarially perturb just the numeric features subject to l2 norm bound
    :param inputs: [n,k] where k is number of
    original input features (PRE-1-hot)
    :param labels: [n,1]
    :param robust: fraction of rows to perturb
    :param indicators: tensor (shape like labels) indicating rows to perturb
    :return: perturbed version of inputs
    '''
    if len(self.numeric_columns) == 0 or robust == 0 or self.l2_epsilon == 0:
      return inputs, None
    coefs = np.array([c[0] for c in self.coefs_by_feature()])
    return RobustLogisticModel.perturb_continuous(inputs, labels, coefs,
                                                  self.l2_epsilon,
                                                  mask=self.numeric_mask,
                                                  perturb_frac=robust,
                                                  indicators=indicators,
                                                  seed=self.seed)


  def perturb_numeric_old(self, x, label, l2):
    '''

    :param x: [n,d] input tensor, pre-1-hot. But note that the
    numeric features may be scatterred at various positions amongst the
    categorical ones. Also note x is NOT in 1-hot form, it will have the
    original  indices of the categorical features.
    :param labels
    :param l2: max l2-norm of perturbation
    :return:
    '''
    coefs = tf.squeeze(tf.gather(self.kernel,
                                 self.numeric_coef_indices))
    n_features = len(self.numeric_columns)
    coefs_l2 = tf.norm(coefs, ord=2)
    x_numeric = x.numpy()[:, self.numeric_columns]
    # the delta that achieves max or min coef * delta
    # subject to l2 norm bound on delta:
    ball_delta = tf.reshape(tf.multiply(l2, tf.divide(coefs, coefs_l2)),
                            [1, -1])
    ball_delta = tf.cast(tf.tile(ball_delta, [x.shape[0], 1]), tf.float64)

    if label == 1:
      ball_delta = -ball_delta

    # Check if ball_delta violates lower/upper bounds (BOX constraints)
    # 1. first get bounds implied on delta from bounds on x + delta

    lbs = np.array(self.lower_bounds) - np.array(x_numeric)
    ubs = np.array(self.upper_bounds) - np.array(x_numeric)

    # ball_delta, lbs, ubs are now: nBatches x nFeatures

    # element-wise indicators (0/1) of whether elements of
    # ball_delta fit within Box constraints
    is_within_bounds = \
      np.less_equal(lbs, ball_delta)*1.0 * \
      np.less_equal(ball_delta, ubs)*1.0

    # nBatches x 1: each element indicates whether
    # the delta row-vector is within bounds
    is_within_bounds = tf.reduce_min(is_within_bounds, axis=1, keep_dims=True)

    is_within_bounds = tf.tile(is_within_bounds, [1, n_features])

    # best "corner" delta perturbation for each input:
    # when Label == 0, we want to maximize coef * delta, so:
    # coef_i = 0 => delta_i = 0 (that dim is immaterial, so keep it 0)
    # coef_i > 0 => delta_i = UB_i (upper bound on dim i)
    # coef_i < 0 => delta_i = LB_i (lower bound on dim i)

    if label == 0:
      best_corner = \
        lbs * np.floor((1.0 - np.sign(coefs))/2.0) + \
        ubs * np.floor((1.0 + np.sign(coefs))/2.0)
    else:
      best_corner = \
        ubs * np.floor((1.0 - np.sign(coefs))/2.0) + \
        lbs * np.floor((1.0 + np.sign(coefs))/2.0)

    # for each input, conditional on is_within_bounds, pick either
    # ball_delta or best_corner

    solution = is_within_bounds *  ball_delta + \
               (1 - is_within_bounds) * best_corner

    x_numeric_perturbed = x_numeric + solution
    loss_change = tf.tensordot(solution, coefs)

    return x_numeric_perturbed, loss_change

  def cat_coefs(self, col_index):
    '''
    All coefs for this categorical column
    :param col_index:
    :return:
    '''
    coef_start_index = self.coef_index_starts[col_index]
    cardinality = self.column_spec[col_index]['card']
    feature_coefs = tf.slice(tf.squeeze(self.kernel),
                             [coef_start_index],
                             [cardinality])
    return feature_coefs.numpy()

  def cat_val_coef(self, col_index, val_index):
    '''
    Get coefficient of categorical feature-value, from current coefs
    :param col_index: index of column in original feature-vector (non 1-hot)
    :param val_index: "value" of feature, i.e. category-index
    :return: coefficient value
    '''
    return self.cat_coefs(col_index)[val_index]

  def cat_worst_coef(self, col_index, label):
    '''
    For the categorical feature at index col_index (pre-1-hot), given the
    label, get the coef (and index) of the worst new value for this categ
    feature.
    :param col_index: index of column in original feature-vector (non 1-hot)
    :param label: 0/1
    :return: worst coef index, and corresponding value
    '''
    coefs = self.cat_coefs(col_index)
    if label == 0:
      return np.argmax(coefs), np.max(coefs)
    else:
      return np.argmin(coefs), np.min(coefs)

  def coefs_col(self, x, i):
    '''
    current coefs  (weights) for a given column of categorical indices
    :param i:
    :return:
    '''
    return tf.reshape(
      tf.gather(self.cat_coefs(i),
                tf.cast(x.numpy()[:, i], tf.int32) ),
      [-1,1]
    )

  def perturb_categorical(self, input, labels, robust=1.0, indicators=None):
    '''
    Given input tensor (PRE-1-hot) nrows x nfeatures, and corresp labels,
    perturb ONLY the categorical features, applying the given l2_epsilon
    constraint to only the categorical features. (later we could have
    separate l2 constraints for categorical and numeric features).

    :param input: batchSize x nInputFeatures (BEFORE 1-hot)
    :param labels: batchSize x 1
    :param robust: fraction of input rows to perturb
    :param indicators: explicit tensor (same shape as labels) of
                       perturbation indicators
    :return: perturbed tensor same shape as input
    '''

    # get the cat, numeric columns separately

    # Notation:
    # variables named "input..." refer to PRE_1-hot,
    # variables named "x..." are POST-1-hot

    if len(self.cat_columns) == 0 or robust == 0:
      return input, None
    # k is the number of categoricals we want to perturb; it is either
    # explicitly given as self.num_perturbed_categoricals,
    # or else
    # k is the biggest int such that 2k <= l2^2, i.e.
    # k = floor( l2_epsilon^2 / 2 ),
    # capped by number of categorical cols
    k = self.num_perturbed_categoricals or int(self.l2_epsilon ** 2 / 2.0)
    k = min(len(self.cat_columns), k)
    if k == 0:
      return input, None

    n_rows = input.shape[0].value
    # categoricals are 1-hot, numeric intact
    x_1_hot = self.one_hot_all(input)
    coefs = tf.reshape(self.kernel, [1,-1])
    x_coefs = x_1_hot * coefs


    input_coefs = self.agg_categoricals(x_coefs, agg_fn=tf.reduce_sum)
    n_col_hot = x_1_hot.shape[1]
    x_all_hot = x_1_hot * 0 + \
                tf.constant(np.repeat(1.0, n_col_hot).astype(np.float32))
    x_all_coefs = x_all_hot * coefs

    if self.config.zap_categoricals:
      input_worst_coefs = input_coefs * 0.0
    else:
      # max coef is worst when label = 0
      input_max_coefs = self.agg_categoricals(x_all_coefs, agg_fn=tf.reduce_max)
      # min coef is worst when label = 1
      input_min_coefs = self.agg_categoricals(x_all_coefs, agg_fn=tf.reduce_min)
      input_worst_coefs = \
        labels * input_min_coefs + (1.0 - labels) * input_max_coefs
    # for each input feature, the worst (absolute) logit-change that
    # can be made by changing that feature-VALUE to a different one.
    # Only non-zero for categoricals; for numerics it will be zero by
    # construction, which is what we want.
    input_potentials = tf.abs(input_worst_coefs - input_coefs)

    # set the potentials of numeric features to -1,
    # to prevent them from being selected in top-k
    block_numerics = self.numeric_mask * (-1.0)
    input_potentials = input_potentials * self.cat_mask + block_numerics

    # In each row pick the biggest k potentials, where

    _, row_indices = tf.nn.top_k(input_potentials, k=k, sorted=False)
    row_indices = tf.cast(tf.expand_dims(row_indices, 2), tf.int64)
    row_nums = tf.tile(tf.reshape(np.arange(n_rows), [-1,1]), [1,k])
    row_nums = tf.expand_dims(row_nums, 2)
    indices_2d = tf.concat([ row_nums, row_indices], axis=2)
    indices_2d = tf.reshape(indices_2d, [-1,2])
    n_indices = indices_2d.shape[0].value
    ones = tf.ones([n_indices])

    # use tf.scatter_nd to create a mask M of 1s corresponding to indices_2d
    # in a zeros tensor of shape same as input_potentials

    perturbed_categoricals_mask = tf.scatter_nd(indices_2d, ones,
                                                input_potentials.shape)
    # multiply by cat_mask to really ensure the numerics are not touched
    perturbed_categoricals_mask = tf.cast(perturbed_categoricals_mask,
                                           tf.float32) * self.cat_mask

    # value-index of max-coef for each input feature
    input_grouped_coefs = self.coefs_by_feature()
    input_max_indices = \
      tf.cast(tf.constant(list(map(np.argmax, input_grouped_coefs))),
              tf.float32)
    input_min_indices = \
      tf.cast(tf.constant(list(map(np.argmin, input_grouped_coefs))),
              tf.float32)
    input_worst_indices = \
      labels * input_min_indices + (1 - labels) * input_max_indices
    # update the input tensor such that wherever the mask M = 1,
    # the feature is changed to the corresponding index in
    # input_worst_indices, and is left intact where M = 0

    if indicators is None:
      indicators = tf.cast(tf.less(tf.random_uniform(tf.shape(labels),
                                                     seed=self.seed),
                                   robust),
                           tf.float32)
    perturbed_categoricals_mask = perturbed_categoricals_mask * indicators
    perturbed_input = perturbed_categoricals_mask * input_worst_indices + \
                      (1.0 - perturbed_categoricals_mask) * tf.cast(input,
                                                                    tf.float32)
    return perturbed_input, indicators


  @staticmethod
  def perturb_continuous(x, labels, coefs, norm_bound,
                         norm_order=2,
                         mask=None, perturb_frac=1.0,
                         indicators=None,
                         seed=None):
    '''
    Generic perturbation with l2 bound and mask
    :param x: [n, d] tensor float32
    :param labels: [n,1] tensor of 0/1 labels
    :param coefs: [d] array of coefs
    :param norm_bound: bound on norm of perturbation
    :param norm_order: order of norm: 2 means L2-norm, -1 means
                            L_inf-norm
    norm of perturbation
    :param mask: [d] 0/1 array of indicators of which dims to perturb
    :param perturb_frac: what fraction to perturb
    :param indicators: explicit tensor (shape like labels) indicating which
           rows to perturb.
    :param seed: random seed for reproducibility
    :return: perturbed version of x
    '''
    d = x.shape[1]
    if perturb_frac == 0 or norm_bound == 0:
      return x, None
    try:
      _, y_shape = labels.get_shape().as_list()
    except Exception as e:
      labels = tf.expand_dims(labels, axis=1)
      pass
    coefs = tf.squeeze(coefs)
    if mask is None:
      mask = np.ones([d])
    coefs = coefs * mask
    if norm_order == 2:
      coefs_l2 = tf.maximum(tf.norm(coefs, ord=2), 1e-8)
      delta = tf.multiply(norm_bound, tf.divide(coefs, coefs_l2))
      delta = (1 - 2 * labels) * delta
    else: # assume L_inf bound
      d = tf_numpy(tf.shape(coefs))[0]
      norm_bound /= math.sqrt(d)
      delta = (1 - 2 * labels) * norm_bound * tf.sign(coefs)
    if indicators is None:
      indicators = tf.cast(tf.less(tf.random_uniform(tf.shape(labels),seed=seed),
                                   perturb_frac), tf.float32)
    return x + delta * indicators, indicators

  def perturb_one_hot_as_continuous(self, x, labels, robust=1.0):
    '''
    Return worst-case adversarial perturbation of x, assuming all continuous
    values, and no box-constraints; so ball-optimum works
    :param x: input tensor (AFTER 1-hot transformation) nrows x d
    :param labels: 0/1 labels tensor, nrows x 1
    :param mask: 0/1 mask of which columns to perturb, shape [d]
    :param robust: fraction of rows of x to be perturbed
    :return:
    '''
    coefs = tf.squeeze(tf.reshape(self.kernel, [1,-1]))
    return RobustLogisticModel.perturb_continuous(x, labels,
                                                  coefs,
                                                  self.l2_epsilon,
                                                  mask=None, perturb_frac=robust,
                                                  seed=self.seed)

  def  attributions(self, input_features, labels, perturb=0.0):
    '''
    Given input batch tensor, get attribution tensor
    :param input_features: input batch: batch_size x n_features (pre-1-hot)
    :param labels: [batch_size, 1]
    :param perturb: what fraction of examples to perturb
    :return: batch_size x n_transformed_features tensor of attributions,
    where n_transformed_features inclues the original numeric features plus
    1-hot encoded categorical features, i.e. the attributions are the level
    of categorical feature-VALUES and numerical features

    '''
    x = self.perturb_encode(input_features, labels, robust=perturb * 1.0)
    bias = self.bias.numpy()[0]
    baseline_prediction = tf.sigmoid(bias).numpy()
    baseline_prediction = tf.tile([[baseline_prediction]],
                                  [input_features.shape[0], 1])
    predictions = self.call(input_features, label=labels, robust=perturb * 1.0)


    # batch_size x n_transformed_features
    net_predictions = tf.tile(predictions - baseline_prediction,
                              [1, x.shape[-1]])

    net_logits = tf.tile(self.logits(input_features, robust=0.0) - bias,
                         [1, x.shape[-1]] )
    logit_contributions = x * tf.transpose(self.kernel)
    attribs = logit_contributions * net_predictions/net_logits
    return attribs

  def agg_categoricals(self, z, agg_fn=tf.reduce_sum):
    '''
    Given a nrows x d feature-tensor (AFTER 1-hot), produce a tensor
    Y (nrows x k) where k is the number of original (PRE-1-hot) features,
    and each categorical feature's column in Y is the aggregation (using
    "agg_fn") of its 1-hot  component columns in z.
    The non-categorical columns in z and Y are identical (i.e. no summing).
    :param z: nrows x d feature-vector (AFTER 1-hot)
    :return: nrows x k, where k is number of original (PRE 1-hot) features
    '''
    nrows = z.shape[0]
    agg_cols = []
    for col in range(self.n_input_features):
      # start, end indices of transformed feature
      start, end = self.coef_index_starts[col], self.coef_index_ends[col]
      col_attribs = agg_fn(
        tf.slice(z, [0, start], [nrows, end - start + 1]),
        axis=1, keepdims=True)
      agg_cols += [ col_attribs ]
    agg_cols = tf.concat(agg_cols, axis=1)
    return agg_cols


  def feature_attributions(self, input_features, labels, perturb=0.0):
    '''
    Similar to attributions but at feature level rather than feature-value
    level (i.e for categoricals, it's for each categorical feature rather
    than feature-value)
    :param input_features:
    :param perturb: what fraction of rows to perturb
    :return:
    '''
    labels = tf.reshape(labels,[-1,1])
    attribs = self.attributions(input_features, labels, perturb=perturb)
    return self.agg_categoricals(attribs, agg_fn=tf.reduce_sum)

  def perturb_encode(self, input, labels, robust=1.0, attacker=None):
    '''
    Perturb AND return 1-hot encoded inputs
    :param input: [?, d] original pre-1-hot features
    :param label: [?, 1]
    :param robust: what fraction to perturb
    :param attacker: optional attacker for perturbation
    :return: perturbed input
    '''
    perturber = attacker or self
    if perturber.config.get('perturb_one_hot', True):
      x = self.one_hot_all(input)
      # Now apply perturbation in TRANSFORMED space!!
      # Treat every dimension as continuous (including 1-hot vecs)
      # NOTE: x will become DENSE since we're not imposing 1-hot constraints,
      # i.e. every feature-value of every feature can in general have a
      # non-zero value, so SGD updates can be SLOW !
      # perturb examples with prob = 'robust'
      x, _ = perturber.perturb_one_hot_as_continuous(x, labels=labels,
                                                  robust = robust)
    else:
      # perturb categoticals, then numericals, in PRE_1-hot stage
      input, indicators = perturber.perturb_categorical(input, labels=labels,
                                            robust=robust)
      input, _  = perturber.perturb_numeric(input, labels=labels,
                                            robust=robust,
                                            indicators=indicators)
      x = self.one_hot_all(input)

    return x


  def logits(self, input, labels=0, robust = 0.0, attacker=None):
    '''
    :param input: nrows x nFeatures (pre-1-hot)
    :param labels: nrows x 1
    :param robust: fraction observations to be perturbed
    :param attacker: (optional) different model to use for perturbations
    :return:
    '''
    # First do feature-transformation
    # replace each categorical column with 1-hot encoded columns.
    x = self.perturb_encode(input, labels=labels, robust=robust,
                            attacker=attacker)
    return tf.add(tf.matmul(x, self.kernel), self.bias)

  def call(self, input, label=0, robust = 0.0, attacker=None):
    return tf.reshape(tf.sigmoid(
      self.logits(input, label, robust, attacker=attacker)), [-1,1])

  def classify(self, input, label=0, robust = 0.0):
    prob = self.call(input, label, robust)
    return 1.0 if prob > 0.5 else 0.0


def grad(model, inputs, targets, robust=0.0,
         attacker:RobustLogisticModel=None):
  with tf.GradientTape() as tape:
    loss_value, _  = loss(model, inputs, targets, robust=robust,
                          attacker=attacker)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def loss(model, x, y, robust=0.0, attacker: RobustLogisticModel=None):
  y = tf.reshape(y, [-1,1])
  y_ = model(x, label=y, robust=robust, attacker=attacker)
  loss_val = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, y_))
  return loss_val, y_
