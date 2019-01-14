#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a custom Estimator for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from src.utils import pmlb_dataset_x_y, tf_entropy, split_and_standardize
from src.utils import tf_feature_columns, df_column_specs
import argparse
import tensorflow as tf
from src.runners import get_run_results
from models.tf_model import binary_classification_model
import shutil
import os
import numpy as np
from src.utils import Bunch
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import pandas as pd
from src.utils import make_sibling_dir, sub_dict
from src.plot import plot_multi
from pmlb import fetch_data
from src.runners import run_5_combos

#TODO Experiment tracking
#TODO Run on our GPU box

plt.interactive(False)


if platform.system() == 'Linux':
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=20, type=int, help='batch size')
parser.add_argument('--eager', dest='eager', action='store_true')
parser.add_argument('--multigpu', dest='multi_gpu', action='store_true')
parser.add_argument('--train_steps', default=None, type=int,
                    help='number of training steps')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of epochs')
parser.add_argument('--dataset', default='credit-a', type=str,
                    help='pmlb dataset name')
parser.add_argument('--log', default='error', type=str,
                    help='pmlb dataset name')

def main(argv):
  args = parser.parse_args(argv[1:])
  log_codes = dict(e = tf.logging.ERROR,
                   i = tf.logging.INFO,
                   w = tf.logging.WARN,
                   d = tf.logging.DEBUG)
  tf.logging.set_verbosity(log_codes.get(args.log.lower()[0],
                                         tf.logging.ERROR))
  dataset_name = args.dataset
  epochs = args.epochs
  batch_size = args.batch_size

  if args.eager:
    tf.enable_eager_execution()
    print('******** TF EAGER MODE ENABLED ***************')

  params = Bunch(perturb_l2_bound=0.2, epochs=epochs, batch_size=batch_size,
                 multi_gpu=args.multi_gpu, lr = 0.01)

  res_10 = get_run_results(dataset_name, params.mod(perturb_frac=0.1))

  res_20 = get_run_results(dataset_name, params.mod(perturb_frac=0.2))

  res_50 = get_run_results(dataset_name, params.mod(perturb_frac=0.5))

  res_80 = get_run_results(dataset_name, params.mod(perturb_frac=0.8))

  res_100 = get_run_results(dataset_name, params.mod(perturb_frac=1.0))

  def get_field(dfs, conditions, column):
    return [df.query(conditions).iloc[0][column] for df in dfs]

  dfs = [res_10, res_20, res_50, res_80, res_100]

  df = pd.DataFrame(dict(
    TrainAdvPct=[10,20,50,80,100],
    TestNat=get_field(dfs, 'train > 0 & test == 0.0', 'acc'),
    TestAdv=get_field(dfs, 'train > 0 & test == 1.0', 'acc')))

  if platform.system() != 'Linux':

    dfm = df.melt('TrainAdvPct', var_name='TestMode', value_name='Accuracy')
    g = sns.factorplot(x="TrainAdvPct", y="Accuracy", hue='TestMode', data=dfm)
    plt.show()


if __name__ == '__main__':
  tf.app.run(main)