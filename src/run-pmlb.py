from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import tensorflow as tf
from src.runners import get_run_results
import os
from src.utils import Bunch, make_clear_dir
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import pandas as pd

plt.interactive(False)


if platform.system() == 'Linux':
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=20, type=int, help='batch size')
parser.add_argument('--eager', dest='eager', action='store_true')
parser.add_argument('--multigpu', dest='multi_gpu', action='store_true')
parser.add_argument('--train_steps', default=None, type=int,
                    help='number of training steps')
parser.add_argument('--epochs', default=400, type=int,
                    help='number of epochs')
parser.add_argument('--dataset', default='credit-a', type=str,
                    help='pmlb dataset name')
parser.add_argument('--log', default='error', type=str,
                    help='pmlb dataset name')
parser.add_argument('--eps', default=0.2, type=float,
                    help='l2_epsilon bound')
parser.add_argument('--norm', default=2, type=int,
                    help='norm order (2 or -1 meaning infinity)')
parser.add_argument('--cont', default=1.0, type=float,
                    help='contamination fraction')



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

  params = Bunch(perturb_norm_bound=args.eps,
                 perturb_norm_order=args.norm,
                 epochs=epochs,
                 batch_size=batch_size, lr=0.01, adv_reg_lambda=0.0,
                 clean_pre_train=0.0,
                 multi_gpu=args.multi_gpu)

  get_run_results(dataset_name, params.mod(perturb_frac=args.cont))

if __name__ == '__main__':
  tf.app.run(main)