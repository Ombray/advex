import tensorflow as tf
import pandas as pd
from src.train import run
from src.utils import Bunch, pmlb_dataset_x_y
tf.enable_eager_execution()
import numpy as np
import yaml

dataset = 'credit-a'
X, y, col_spec, target = pmlb_dataset_x_y(dataset)

np.random.seed(123)

config = Bunch(batch_size=20,
               perturb_one_hot=False,
               zap_categoricals=False,
               l2_epsilon=0.0,
               num_perturbed_categoricals=1,
               col_spec=col_spec,
               label_name=target)

pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 1000)
pd.set_option('display.width', 1000)
epochs = 400
batch = 20
lr = 0.01

res_10 = run(config, X, y,
             robust_frac=0.1,
             epochs=epochs,
             batch_size=batch,
             lr=lr,
             verbose=False)

res_20 = run(config, X, y,
             robust_frac=0.2,
             epochs=epochs,
             batch_size=batch,
             lr=lr,
             verbose=False)


res_50 = run(config, X, y,
             robust_frac=0.5,
             epochs=epochs,
             batch_size=batch,
             lr=lr,
             verbose=False)

res_80 = run(config, X, y,
             robust_frac=0.8,
             epochs=epochs,
             batch_size=batch,
             lr=lr,
             verbose=False)

res_100 = run(config, X, y,
              robust_frac=1.0,
              epochs=epochs,
              batch_size=batch,
              lr=lr,
              verbose=False)

cols =  [
  'train', 'test', 'l2',
  'train_loss',
  'loss', 'auc', 'acc', 'r2',
  'bias', 'ent',
  # 'coefs'
  # 'attr_ave', 'attr_abs'
]

print('perturb-frac = 0.1')
print(res_10[cols])
print('perturb-frac = 0.2')
print(res_20[cols])
print('perturb-frac = 0.5')
print(res_50[cols])
print('perturb-frac = 0.8')
print(res_80[cols])
print('perturb-frac = 1.0')
print(res_100[cols])

def get_field(dfs, conditions, column):
  return [df.query(conditions).iloc[0][column] for df in dfs]

dfs = [res_10, res_20, res_50, res_80, res_100]

df = pd.DataFrame(dict(
  TrainAdvPct=[10,20,50,80,100],
  TestNat=get_field(dfs, 'train > 0 & test == 0.0', 'acc'),
  TestAdv=get_field(dfs, 'train > 0 & test == 1.0', 'acc')))

import matplotlib.pyplot as plt
import seaborn as sns
plt.interactive(False)

dfm = df.melt('TrainAdvPct', var_name='TestMode', value_name='Accuracy')
g = sns.factorplot(x="TrainAdvPct", y="Accuracy", hue='TestMode', data=dfm)
plt.show()

