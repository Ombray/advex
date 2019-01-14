from data_loader.data_generator import DataGenerator
import pandas as pd
import tensorflow as tf
from src.utils import randnsphere, Bunch




class CSVData(DataGenerator):
  def __init__(self, config: Bunch):
    self.config = config
    col_names = [c.get('name', None) for c in config.col_spec]
    if None in col_names: # get them from file header
      col_names = None
    dataset = tf.contrib.data.make_csv_dataset(
        config.file,
        config.batch_size,
        column_names=col_names, # assume 1st row is header row
        label_name=config.label_name,
        shuffle=True,
        num_epochs=1) # forever
    dataset = dataset.map(DataGenerator.pack_features_vector)
    self.dataset = dataset.map(self.normalize)


  def normalize(self, feats, labels):
    if self.config.get('means'):
      feats = feats - self.config.means
    if self.config.get('stdevs'):
      feats = feats/self.config.stdevs
    return feats, labels

  def pack_features(self, feat_dict, labels):
    feats = []
    for spec in self.config.col_spec:
      if spec['name'] == self.config.label_name:
        continue
      col = feat_dict[spec['name']]
      if spec['type'] == 'num':
        feats += [ tf.reshape(col,[-1,1]) ]
      else:
        col = tf.cast(col, tf.int32)
        feats += [ tf.one_hot(col, spec['card']) ]
    feats = tf.concat(list(feats), axis=1)
    return feats, tf.reshape(tf.cast(labels, tf.float32),[-1,1])

  def perturb(self, dataset, l2_epsilon=0.0):
    '''
    CAUTION - this fails when passed in raw non-1-hot data;
    Anyway we don't need non-adversarial (random) perturbations for now.

    Random, NOT adversarial epsilon-perturbation, hence
    it doesn't depend on any model
    :param dataset:
    :param l2_epsilon:
    :return:
    '''
    def _perturb(features, labels):
      normals = tf.random_normal(tf.shape(features))
      units = tf.nn.l2_normalize(normals, axis=1)
      return features + units * l2_epsilon, labels

    return dataset.map(_perturb)

  def data(self, l2_epsilon=0.0, take=0, skip=0):
    ds = self.dataset
    if take > 0:
      ds = ds.take(take)
    if skip > 0:
      ds = ds.skip(skip)
    #ds = self.perturb(ds, l2_epsilon)
    return ds




