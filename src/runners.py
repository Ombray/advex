from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from src.utils import tf_entropy, split_and_standardize, make_sibling_dir
from src.utils import make_clear_dir, Bunch
from src.utils import tf_feature_columns, df_column_specs, sub_dict, tf_numpy
from src import utils
from src.plot import plot_multi
import tensorflow as tf
from data_loader.synth_gen import gen_synth_df
from models.tf_model import binary_classification_model
import os
import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.model_selection import ParameterGrid
#from test_tube import Experiment
import platform

# modes

class Runner:
  def __init__(self, dataset_name, df: pd.DataFrame, model_dir: str,
               params: Bunch = None):
    self.col_spec, self.target_name = df_column_specs(df)

    self.segments = np.array([], dtype=np.int32)
    self.feature_value_names = []
    for i, s in enumerate(self.col_spec):
      self.segments = np.append(self.segments, [np.repeat(i, s['card'])])
      col_name = s['name']
      if s['card'] == 1:
        self.feature_value_names += [col_name]
      else:
        self.feature_value_names += [col_name + '=' + str(i)
                                     for i in range(s['card'])]

    # prepend numeric to enforce lexicographic order so
    # we can recover the model weights from tensorflow's variables
    # in the right order
    df = df.copy(deep=True)
    df.columns = [c if c == self.target_name else f'{i:05d}_' + c \
                  for i, c in enumerate(df.columns)]

    self.feature_columns = tf_feature_columns(df)
    self.df_train, self.df_test = \
      split_and_standardize(df, params.get('std', True))

    dir = make_sibling_dir(__file__,
                           f'datasets/{dataset_name}')
    self.train_file = f'{dir}/train.csv'
    self.test_file = f'{dir}/test.csv'

    with open(self.train_file, mode='w+') as fd:
      self.df_train.to_csv(fd, header=True, index=False)

    with open(self.test_file, mode='w+') as fd:
      self.df_test.to_csv(fd, header=True, index=False)

    self.model_dir = model_dir

  def train(self, params: Bunch):
    tf.set_random_seed(123)
    np.random.seed(123)

    config = None
    if platform.system() == 'Linux' and multi_gpu: # we're on the GPU
      # Thanks to
      # https://medium.com/tensorflow/multi-gpu-training-with-estimators-tf-keras-and-tf-data-ba584c3134db
      NUM_GPUS = 10
      strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
      config = tf.estimator.RunConfig(train_distribute=strategy)

    classifier = tf.estimator.Estimator(
      model_fn=binary_classification_model,
      model_dir=self.model_dir,
      config=config,
      params=params.mod(
        segments=self.segments,
        feature_columns=self.feature_columns,
        train_perturb_frac=params.perturb_frac,
        test_perturb_frac=0.0))

    def input_fn():
      df_target = self.df_train[self.target_name]
      ds = tf.data.Dataset.from_tensor_slices((dict(self.df_train), df_target))
      ds = ds.shuffle(buffer_size=1000, seed=123).\
        repeat(params.epochs).prefetch(params.batch_size*3).\
        batch(params.batch_size)
      if platform.system() == 'Linux':
        ds = ds.apply(tf.contrib.data.prefetch_to_device(
          device='/device:gpu:0'))
      return ds

    # define train input_fn
    # input_fn = lambda: tf.contrib.data.make_csv_dataset(
    #   self.train_file,
    #   batch_size=batch_size,
    #   num_epochs=epochs,
    #   shuffle=True,
    #   shuffle_buffer_size=10000,
    #   shuffle_seed=123,
    #   prefetch_buffer_size=batch_size*3,
    #   num_parallel_reads=10,
    #   label_name=self.target_name
    # )

    # Tried to use this to show train/test loss at each epoch but
    # doesn't work as there's an outstanding bug-fix in TF.
    # evaluator = tf.contrib.estimator.InMemoryEvaluatorHook(
    #   classifier, eval_input_fn, steps=None, every_n_iter=100)

    # Train the Model.
    classifier.train(input_fn=input_fn,  # hooks=[evaluator],
                     steps=None)
    return classifier

  def eval(self, params: Bunch):
    tf.set_random_seed(123)
    np.random.seed(123)

    classifier = tf.estimator.Estimator(
      model_fn=binary_classification_model,
      model_dir=self.model_dir,
      config=tf.estimator.RunConfig(tf_random_seed=123),
      params=params.mod(
        segments=self.segments,
        feature_columns=self.feature_columns,
        train_perturb_frac=0.0,
        test_perturb_frac=params.perturb_frac))

    def input_fn(train=False):
      df = self.df_train if train else self.df_test
      df_target = df[self.target_name]
      ds = tf.data.Dataset.from_tensor_slices((dict(df), df_target)).\
        prefetch(params.batch_size*3).\
        batch(params.batch_size)
      if platform.system() == 'Linux':
        ds = ds.apply(tf.contrib.data.prefetch_to_device(
          device='/device:gpu:0'))
      return ds

    # input_fn = lambda: tf.contrib.data.make_csv_dataset(
    #   self.test_file,
    #   batch_size=batch_size,
    #   prefetch_buffer_size=batch_size*3,
    #   num_parallel_reads=10,
    #   num_epochs=1,
    #   shuffle=False,
    #   label_name=self.target_name
    # )

    eval_result = classifier.evaluate(input_fn=lambda: input_fn(train=False))
    train_result = classifier.evaluate(input_fn=lambda: input_fn(train=True))

    # Note we are computing ALL attribution metrics on
    # UNPERTURBED TRAIN data !
    feature_value_attribs = train_result['attrib']

    feature_value_attribs_ent = tf_entropy(feature_value_attribs)
    feature_attribs = tf.segment_sum(feature_value_attribs, self.segments)
    feature_attribs_ent = tf_entropy(feature_attribs)

    afvi = train_result['afvi']
    afvi_ent = tf_entropy(afvi)
    feature_afvi = tf.segment_sum(afvi, self.segments)
    feature_afvi_ent = tf_entropy(feature_afvi)

    col_names = [s['name'] for s in self.col_spec]
    feature_attribs_dict = dict(zip(col_names,
                                    np.round(tf_numpy(feature_attribs),
                                             4)))
    feature_afvi_dict = dict(zip(col_names,
                                 np.round(tf_numpy(feature_afvi), 4)))
    wts = classifier.get_variable_value('dense/kernel').squeeze()
    wts_dict = dict(zip(self.feature_value_names, wts))

    feat_label_corrs = \
      (train_result['xy_av'] - train_result['x_av'] * train_result['y_av']) / \
      np.sqrt( (train_result['xsq_av'] - train_result['x_av']**2) * \
               (train_result['ysq_av'] - train_result['y_av']**2) )

    corrs_dict = dict(zip(self.feature_value_names, feat_label_corrs))

    wts_ent = tf_numpy(tf_entropy(wts))
    wts_l1 = tf_numpy(tf.norm(wts, ord=1))
    wts_max = np.max(np.abs(wts))
    wts_l1_linf = wts_l1 / wts_max
    wts_1pct = np.sum(np.abs(wts) > 0.01 * wts_max)
    wts_pct1pct = 100*np.sum(np.abs(wts) > 0.01 * wts_max)/len(wts)
    results = dict(
      acc=np.round(eval_result['accuracy'], 3),
      auc=np.round(eval_result['auc'], 3),
      loss=np.round(eval_result['loss'], 3),
      wts_ent=np.round(wts_ent,3),
      wts_1pct=wts_1pct,
      wts_pct1pct=wts_pct1pct,
      wts_l1=np.round(wts_l1,3),
      wts_l1_linf=np.round(wts_l1_linf,3),
      av_ent=np.round(train_result['attrib_ent'], 7),
      av_high=np.round(train_result['high_attribs'], 1),
      a_ent=np.round(tf_numpy(afvi_ent), 3),
      f_a_ent=np.round(tf_numpy(feature_afvi_ent), 3),
      g_ent=np.round(tf_numpy(feature_value_attribs_ent), 3),
      f_g_ent=np.round(tf_numpy(feature_attribs_ent), 3),
      f_a_dict=feature_afvi_dict,
      f_g_dict=feature_attribs_dict,
      wts_dict=wts_dict,
      corrs_dict=corrs_dict
    )
    return results


def run_5_combos(dataset_name, df, model_dir, params: Bunch):

  runner = Runner(dataset_name, df, model_dir, params)
  make_clear_dir(model_dir)
  # exp_dir = make_sibling_dir(__file__, 'experiments')
  # exp = Experiment(name=dataset_name, debug=False,  save_dir=exp_dir)
  # exp.tag(params)

  # natural training
  runner.train(params.mod(perturb_frac=0.0))
  nat_nat = runner.eval(params.mod(perturb_frac=0.0))
  nat_nat.update(train=0.0, test=0.0)
  # exp.log(nat_nat)
  printed_cols = list(set(nat_nat.keys()).difference(['f_a_dict', 'f_g_dict',
                                                 'wts_dict', 'corrs_dict']))
  print(pd.DataFrame([nat_nat])[printed_cols])


  nat_per = runner.eval(params)
  nat_per.update(train=0.0, test=params.perturb_frac)
  # exp.log(nat_per)
  print(pd.DataFrame([nat_per])[printed_cols])

  # adversarial training: Start with naturally trained classifier
  # for epochs/2, then train on adversarial inputs for remaining
  # epochs/2
  make_clear_dir(model_dir)
  clean_train_epochs = int(params.get('clean_pre_train', 0.5) * params.epochs)
  dirty_train_epochs = params.epochs - clean_train_epochs
  if clean_train_epochs > 0:
    runner.train(params.mod(perturb_frac=0.0, epochs=clean_train_epochs))
  runner.train(params.mod(epochs=dirty_train_epochs))

  per_nat = runner.eval(params.mod(perturb_frac=0.0))
  per_nat.update(train=params.perturb_frac, test=0.0)
  # exp.log(per_nat)
  print(pd.DataFrame([per_nat])[printed_cols])

  per_per = runner.eval(params)
  per_per.update(train=params.perturb_frac, test=params.perturb_frac)
  # exp.log(per_per)
  print(pd.DataFrame([per_per])[printed_cols])

  per_per_all = runner.eval(params.mod(perturb_frac=1.0))
  per_per_all.update(train=params.perturb_frac, test=1.0)
  # exp.log(per_per_all)
  print(pd.DataFrame([per_per_all])[printed_cols])

  all_results = dict(
    nat_nat=nat_nat,
    nat_per=nat_per,
    per_nat=per_nat,
    per_per=per_per,
    per_per_all=per_per_all)

  return all_results

def run_one(df: pd.DataFrame, params: Bunch, name='one'):
  '''
  Only do a single train (nat or adv) and test (nat or adv) combo
  and return some metrics/values
  :param df:
  :param params:
  :return:
  '''

  tf.set_random_seed(123)
  np.random.seed(123)
  model_dir = os.path.join('/tmp/robulin/exp', name)

  runner = Runner(name, df, model_dir, params)
  make_clear_dir(model_dir)

  # adversarial training: pre-train on nat examples for
  # some fraction of epochs, then on adversarial for remaining epochs
  clean_train_epochs = int(params.get('clean_pre_train', 0.5) * params.epochs)
  dirty_train_epochs = params.epochs - clean_train_epochs
  if clean_train_epochs > 0:
    runner.train(params.mod(perturb_frac=0.0,
                            epochs=clean_train_epochs))
  runner.train(params.mod(epochs=dirty_train_epochs))
  result = runner.eval(params.mod(
    perturb_frac=params.test_perturb_frac))
  result.update(train=params.perturb_frac, test=params.test_perturb_frac)

  few_fields = params.get('fields',
                          ['train', 'test', 'loss', 'auc', 'acc',
                          'wts_ent', 'wts_l1', 'wts_l1_linf', 'wts_1pct',
                           'wts_pct1pct',
                          'av_ent', 'av_high',
                          'a_ent',  'g_ent',
                          'f_a_ent', 'f_g_ent' ])

  result_few = sub_dict(result, few_fields)
  simple_keys = [k for k,v in params.items()
                 if type(v) in [int, float, str, bool]]
  result_few.update(sub_dict(params, simple_keys))
  print(result_few)
  result.update(sub_dict(params, simple_keys))
  return result

def run_grid(params:Bunch):
  '''
  Run a grid of experiments based on params.grid and return collated
  values/metrics in a data-frame
  :param params:
  :return:
  '''
  if params.get('dataset'):
    df = fetch_data(params.dataset)
  else:
    df = gen_synth_df(params)

  grid_dict = params.grid
  params_list = list(ParameterGrid(grid_dict))
  results = []
  for p in params_list:
    result = run_one(df, params.mod(p), name=params.get('dataset', 'synth'))
    results += [result]
  results = pd.DataFrame(results)
  return results




def get_run_results(dataset_name, params: Bunch, df=None):
  pd.set_option('display.max_columns', 500)
  pd.set_option('max_colwidth', 1000)
  pd.set_option('display.width', 1000)

  tf.set_random_seed(123)
  np.random.seed(123)
  model_dir = os.path.join('/tmp/robulin', dataset_name)

  # fetch a PMLB dataset as a data-frame
  if df is None:
    df = fetch_data(dataset_name)
  results = run_5_combos(dataset_name, df, model_dir, params)
  perturb = params.perturb_frac

  results_dir = make_sibling_dir(__file__, f'results/pert={perturb}')

  few_fields = ['train', 'test', 'loss', 'auc', 'acc',
                'wts_ent', 'wts_l1', 'wts_l1_linf', 'wts_1pct',
                'wts_pct1pct',
                'av_ent', 'av_high',
                'a_ent',  'g_ent',
                'f_a_ent', 'f_g_ent' ]

  keys = ['nat_nat', 'nat_per', 'per_nat', 'per_per', 'per_per_all']
  results_few = pd.DataFrame([ sub_dict(results[k], few_fields)
                               for k in keys ])[few_fields]

  with open(os.path.join(results_dir, 'summary.csv'), 'w+') as fd:
    results_few.to_csv(fd, float_format='%.3f', index=False)

  print(results_few)

  # show nat_nat and per_nat IG attribs
  attr_nat = pd.DataFrame([results['nat_nat']['f_g_dict']]).transpose()
  attr_per = pd.DataFrame([results['per_nat']['f_g_dict']]).transpose()
  attribs: pd.DataFrame = pd.concat([attr_nat, attr_per], axis=1)
  attribs.columns = ['nat', 'adv']
  attribs['feature'] = attribs.index
  print(f'IG Attribs: nat_nat vs per_nat')
  print(attribs.sort_values(by='nat', ascending=False))
  if platform.system() != 'Linux':
    plot_multi(attribs, 'feature', value='attrib', order_by='nat',
               var='train mode')

  # show nat_nat and per_nat wts
  wts_nat = pd.DataFrame([results['nat_nat']['wts_dict']]).transpose()
  wts_per = pd.DataFrame([results['per_nat']['wts_dict']]).transpose()
  wts: pd.DataFrame = pd.concat([wts_nat, wts_per], axis=1)
  wts.columns = ['nat', 'adv']
  wts['feature'] = wts.index
  wts['nat_abs'] = abs(wts['nat'])
  print(f'Wts: nat_nat vs per_nat')
  print(wts.sort_values(by='nat_abs', ascending=False))
  if platform.system() != 'Linux':
    plot_multi(wts, 'feature', value='wt', order_by='nat_abs',
               ignore=['nat_abs'], var='train mode')

  # save various results
  for k in keys:
    dir = make_sibling_dir(__file__, f'results/{dataset_name}/'
                                     f'pert={perturb}/attrib/{k}')
    ig_dict = pd.DataFrame([results[k]['f_g_dict']]).transpose()
    afvi_dict = pd.DataFrame([results[k]['f_a_dict']]).transpose()
    attribs = pd.concat([ig_dict, afvi_dict], axis=1)
    attribs.columns = ['ig', 'afvi']
    # if k in ['nat_nat', 'per_nat']:
    #   print(f'attrib for {k}:')
    #   print(attribs)
    #   attribs['feature'] = attribs.index
    #   if platform.system() != 'Linux':
    #     plot_multibar(attribs, 'feature', value='attrib', order_by='ig')

    with open(os.path.join(dir, 'attrib.csv'), 'w+') as fd:
      attribs.to_csv(fd)

  tf.logging.info('*** tensorboard cmd:')
  tf.logging.info(f'tensorboard --logdir={model_dir}')

  return results_few
