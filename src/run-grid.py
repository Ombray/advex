from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import tensorflow as tf
from src.runners import get_run_results, run_grid
import os
from src.utils import Bunch, make_sibling_dir, df_simple, sub_dict_prefix
import matplotlib.pyplot as plt
import yaml
import numpy as np
from data_loader.synth_gen import gen_synth_df
import seaborn as sns
import platform
import pandas as pd
import math
import time

plt.interactive(False)

pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 1000)
pd.set_option('display.width', 1000)

if platform.system() == 'Linux':
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--hparams', default='hparams/synth.yaml', type=str,
                    help='params grid to run exps on')
parser.add_argument('--out', type=str, help='results output file')

def main(argv):
  args = parser.parse_args(argv[1:])
  log_codes = dict(e = tf.logging.ERROR,
                   i = tf.logging.INFO,
                   w = tf.logging.WARN,
                   d = tf.logging.DEBUG)

  params = Bunch(yaml.load(open(args.hparams)))

  tf.logging.set_verbosity(log_codes.get(params.log.lower()[0],
                                         tf.logging.ERROR))



  if params.eager:
    tf.enable_eager_execution()
    print('******** TF EAGER MODE ENABLED ***************')

  timestr = time.strftime("%Y%m%d-%H%M%S")
  default_out_file_base =  os.path.join(
    os.path.basename(args.hparams).split('.')[0],
    timestr)
  out_file_base = args.out or default_out_file_base

  results = run_grid(params)

  if not params.get('dataset'): # i.e if synthetic, not UCI/pmlb
    results['Wts_x_L1'] = results['wts_dict']. \
      map(lambda w: np.sum(np.abs(np.array(
          list(sub_dict_prefix(w, 'x').values())))))

    results['Wts_r_L1'] = results['wts_dict']. \
      map(lambda w: np.sum(np.abs(np.array(
          list(sub_dict_prefix(w, 'r').values())))))

    results['Wts_w_L1'] = results['wts_dict']. \
      map(lambda w: np.sum(np.abs(np.array(
      list(sub_dict_prefix(w, 'w').values())))))

  results_simple = results.drop(['wts_dict', 'f_g_dict', 'f_a_dict'], axis=1)

  results_dir = make_sibling_dir(__file__, 'results')
  results_simple_file = os.path.join(results_dir, out_file_base + '.csv')
  os.makedirs(os.path.dirname(results_simple_file), exist_ok=True)
  with open(results_simple_file, 'w+') as fd:
    results_simple.to_csv(fd, float_format='%.3f', index=False)

  pkl_file = os.path.join(results_dir, out_file_base + '.pkl')
  results.to_pickle(pkl_file)
  print(df_simple(results))

  print(f'******** Summary csv written to {results_simple_file}')
  print(f'******** Pickled results dataframe at {pkl_file}')

if __name__ == '__main__':
  tf.app.run(main)