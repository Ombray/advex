import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from src.utils import Bunch
from data_loader.data_generator import DataGenerator

def make_cat(card, n=1):
  return np.random.choice(card, n) * 1.0


def make_num(n=1):
  return np.random.normal(0.0, 1.0, n)



class SyntheticDataGenerator(DataGenerator):
    def __init__(self, config:Bunch=None):
      if not config:
        config = Bunch(N = 5000, batch_size=50,
                       col_spec = [
                       dict(type='cat', card=3),
                       dict(type='cat', card=5),
                       dict(type='num', min=0, max=6),
                       dict(type='num', min=-3, max=3)],
                       noise = 0.4,
                       coefs=np.array([  -1.0, 1.0, 0.4, # cat feature 1
                         -1.0, 3.0, 0.9, -1.0, -0.5, # cat feature 2
                         0.8,  # n1
                         -0.9])
                       )
      self.config = config
      N = config.N
      columns = []
      for spec in config.col_spec:
        if spec['type'] == 'cat':
          columns += [make_cat(spec['card'], N)]
        else:
          columns += [make_num(N)]

      arr = np.array(columns).transpose()
      cat_features = [i for i, spec in enumerate(config.col_spec)
                      if spec['type'] == 'cat']
      arr_hot = arr
      if len(cat_features) > 0:
        enc = OneHotEncoder(categorical_features=cat_features)
        enc.fit(arr)
        arr_hot = enc.transform(arr).toarray()

      logits = np.matmul(arr_hot, config.coefs) + config.bias

      noise = np.random.normal(0.0, config.noise, size=N)
      logits += noise

      probs = 1.0/(1 + np.exp(-logits))
      labels = np.array([np.random.choice(2, 1, p=[1-p, p])[0] for p in
                         probs]
                        ).reshape([-1,1])

      # features = df[['c1', 'c2', 'n1', 'n2']].to_dict('list')
      # labels = df[['label']].to_dict('list')
      self.X = tf.cast(arr, tf.float32)
      self.Y = tf.cast(labels, tf.float32)






