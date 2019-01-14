from src.utils import Bunch
import pandas as pd
import numpy as np
import math

def gen_synth_df(params: Bunch):
  N = params.get('N', 1000)
  nC = params.get('nc', 8) # number of features predictive of label
  nF = params.get('nr', 8) # how many features are random, uncorr with
  nW = params.get('nw', 0) # how many features weakly correlated with label
  corr1 = params.get('corrp', True) # whether predictive feats are corr
  corr = params.get('corr', False) # whether random feats are corr
  cat = params.get('cat', False) # whether the random features are categorical
  pred = params.get('pred', 0.7) # how often do the "predictive" features
  weak_pred = params.get('weak_pred', 1.0) # predictivity of weak feature

  np.random.seed(123)

  y = np.random.choice(2, N, p = [0.5, 0.5])
  y1 = (2*y - 1.0).reshape([-1,1])
  df_agree = pd.DataFrame()
  p = pred # probability of agreement with label
  if nC > 0:
    if corr1: # identical, i.e. highly correlated
      agree = np.repeat(
        np.array(np.random.choice(2, N, p=[1-p, p]), dtype=np.float32). \
          reshape([-1,1]), nC, axis=1)
    else: # uncorrelated
      agree = np.array(np.random.choice(2, N * nC, p=[1-p, p]),
                       dtype=np.float32). \
        reshape([-1, nC])
    agree = y1*(2*agree-1)
    agree_cols = ['x' + str(i+1) for i in range(nC)]
    df_agree = pd.DataFrame(agree, columns=agree_cols)

  #
  df_tar = pd.DataFrame(dict(target=y))

  # rest are random
  df_rest = pd.DataFrame()
  if nF > 0:
    if cat: # 1 categorical feature with nF possible values
      rest = np.array(np.random.choice(nF, N), dtype=np.int64). \
        reshape([-1,1])
    else:
      if corr: # identical, i.e. highly correlated
        rest = np.repeat(
          np.array(np.random.choice(2, N, p=[0.5, 0.5]), dtype=np.float32). \
            reshape([-1,1]), nF, axis=1)
      else: # uncorrelated, i.i.d -1/1
        rest = np.array(np.random.choice(2, N * nF, p=[0.5, 0.5]),
                        dtype=np.float32).reshape([-1, nF])
      rest = 2*rest - 1.0
    rest_cols = ['r'] if cat else ['r' + str(i+1) for i in range(nF)]
    df_rest = pd.DataFrame(rest, columns=rest_cols)

  df_weak = pd.DataFrame()
  if nW > 0: # normal, slightly correlated with label
    means = y1 * weak_pred / math.sqrt(nW)
    weak = np.repeat(np.random.normal(means, 1.0), nW, axis=1)
    weak_cols = ['w'] if cat else ['w' + str(i+1) for i in range(nW)]
    df_weak = pd.DataFrame(weak, columns=weak_cols)

  df = pd.concat([df_agree, df_rest, df_weak, df_tar], axis=1)
  return df
