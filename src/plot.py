import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.utils import Bunch
from src.utils import last_file


def plot_multi(df: pd.DataFrame, x_ids=None, var='type',
               value='value', include=None, ignore=None,
               order_by=None, ascending=False, title=None,
               rename = dict(), legend=True,
               threshold=0.001, kind='bar', show=True,
               tight=True,
               ax=None,
               params: Bunch = Bunch()):
  # rcParams.update({'figure.autolayout': True})
  if rename:
    df = df.rename(columns=rename)
  if x_ids is None:
    df['__x'] = df.index
    x_ids = '__x'
  if order_by is not None:
    df = df.sort_values(by=order_by, ascending=ascending)
  if ignore:
    df = df.drop(ignore, axis=1)
  if include:
    df = df[include]
  plt.interactive(False)
  dfm = pd.melt(df, id_vars=x_ids, var_name=var, value_name=value)
  biggest = np.max(np.abs(dfm[value]))
  dfm = dfm[abs(dfm[value]) >= biggest * threshold]
  x_labels = df[x_ids].astype(str).tolist()
  x_labels = [x for x in x_labels if x in dfm[x_ids].astype(str).tolist()]
  max_x_label = max( [ len(s) for s in x_labels])
  if kind=='bar':
    g = sns.catplot(x=x_ids, y=value, hue=var, data=dfm, kind=kind,
                    order=x_labels,
                    row_order=x_labels,
                    legend=False, legend_out=True, ax=ax)
    if params.get('xtick_font_size'):
      g.set_xticklabels(fontdict=dict(fontsize=params.xtick_font_size))
    if max_x_label > 4:
      g.set_xticklabels(rotation=90)
  else: # line
    g = sns.lineplot(x=x_ids, y=value, hue=var, data=dfm,
                     legend=False, ax=ax)
    #g.set_xticklabels(labels=x_labels)
  if title:
    plt.title(title)
  if tight:
    plt.tight_layout()
  if legend:
    plt.legend(loc='upper right', title=var)
  if show:
    plt.show()
  return g


def plot_drop_arv(df: pd.DataFrame, x_ids=None, var='type',
              value='value', ignore=None,
              threshold=0.01, prefix=''):
  if x_ids is None:
    df['__x'] = df.index.astype(str)
    x_ids = '__x'
  if prefix != '':
    df = df[df[x_ids].str.startswith(prefix)]
  var_cols = list(df.columns)
  var_cols.remove(x_ids)
  col0 = np.abs(np.array(df[var_cols[0]]))
  col1 = np.abs(np.array(df[var_cols[1]]))
  col0[::-1].sort()
  col1[::-1].sort()
  df[var_cols[0]] = col0
  df[var_cols[1]] = col1
  df.index = range(len(df))
  pct = 100*np.array(df.index)/len(df)
  df['pct'] = pct
  df = df.drop(x_ids, axis=1)
  df = df.sort_values(by=var_cols[0], ascending=False)
  if ignore:
    df = df.drop(ignore, axis=1)
  plt.interactive(False)
  dfm = pd.melt(df, id_vars='pct', var_name=var, value_name=value)
  if threshold > 0:
    biggest = np.max(np.abs(dfm[value]))
    dfm = dfm[abs(dfm[value]) > biggest * threshold]
  g = sns.lineplot(x='pct', y=value, hue=var, data=dfm, )
  title = 'abs feature weights by percentile'
  if prefix != '':
    title = title + f' for prefix {prefix}'
  plt.title(title)


def df_dict_cols(df: pd.DataFrame, field,
                 names=['r1', 'r2'], rows=[0,1],
                 sort = None, dist = 0.0, absval=False,
                 index='features', prefix=None):
  if sort:
    df = df.sort_values(by=sort, ascending=False)
    highest = df[sort].max(axis=0)
    df = df[df[sort] >= highest - dist]
    df = df.iloc[[0,-1], :] # firt and last

  dicts = list(df[field])
  dict1 = dicts[rows[0]]
  dict2 = dicts[rows[1]]
  df1 = pd.DataFrame([dict1]).transpose()
  df2 = pd.DataFrame([dict2]).transpose()
  df_: pd.DataFrame = pd.concat([df1, df2], axis=1)
  if absval:
    df_ = abs(df_)
  df_.columns = names

  x_ids = index
  df_[x_ids] = df_.index

  if prefix:
    df_ = df_[df_[x_ids].str.startswith(prefix)]

  return df_


def plot_drop(params: Bunch):

  file = last_file(params.file, pattern="*pkl")
  df = df_dict_cols(pd.read_pickle(file),
                    field=params.field,
                    names=params.names,
                    rows=params.rows,
                    sort=params.sort,
                    dist=params.dist,
                    prefix=params.prefix,
                    index=params.index)

  var_cols = list(df.columns)
  x_ids = params.index
  var_cols.remove(x_ids)
  col0 = np.abs(np.array(df[var_cols[0]]))
  col1 = np.abs(np.array(df[var_cols[1]]))
  col0[::-1].sort()
  col1[::-1].sort()
  df[var_cols[0]] = col0
  df[var_cols[1]] = col1
  df.index = range(len(df))
  pct = 100*np.array(df.index)/len(df)
  df['pct'] = pct
  df = df.drop(x_ids, axis=1)
  df = df.sort_values(by=var_cols[0], ascending=False)
  if params.ignore:
    df = df.drop(params.ignore, axis=1)
  plt.interactive(False)
  dfm = pd.melt(df, id_vars='pct', var_name=params.var,
                value_name=params.value)
  if params.threshold > 0:
    biggest = np.max(np.abs(dfm[params.value]))
    dfm = dfm[abs(dfm[params.value]) > biggest * params.threshold]
  g = sns.lineplot(x='pct', y=params.value, hue=params.var, data=dfm)
  title = params.title
  if params.prefix:
    title = title + f' for prefix {params.prefix}'
  plt.title(title)
  plt.show()

def plot_bars(params: Bunch):
  # (pkl_file, field = 'f_g_dict',
  #           names = ['nat', 'adv'], ylabel='Value',  title='value'):
  file = last_file(params.file, pattern="*pkl")
  df = df_dict_cols(pd.read_pickle(file),
                      field=params.field,
                      names=params.names,
                      rows=params.rows,
                      sort=params.sort,
                      dist=params.dist,
                      absval=params.absval,
                      prefix=params.prefix,
                      index=params.index)

  print(f'{params.title}: {params.names[0]} vs {params.names[1]}')
  print(df.sort_values(by=params.names[0], ascending=False))
  plot_multi(df, params.index, value=params.ylabel, order_by=params.names[0],
             ascending=False, title=params.title, threshold=params.threshold,
             var=params.var, params=params)


def plot_vars(file, params):
  x_var = params.x_var
  x_descr = params.x_descr
  band_min = params.band_min
  band_max = params.band_max
  ylabels = params.ylabels
  xlabel = params.xlabel
  vars = params.vars

  file = last_file(file, pattern="*pkl")
  df = pd.read_pickle(file)
  fig, axes = plt.subplots(nrows=len(vars), ncols=1, squeeze=True)
  #fig.tight_layout(pad=0, h_pad=0, w_pad=0)

  fig.subplots_adjust(hspace = 0.0, wspace=0.0)

  var1 = plot_multi(df,
                   rename=dict(perturb_norm_bound='epsilon'),
                   #x_ids = 'epsilon',
                   x_ids = x_var,
                   var='',
                   legend=False,
                   value=ylabels[0],
                   include = [x_var, vars[0]],
                   order_by=x_var, ascending=True,
                   threshold=0.0,
                    tight=False,
                   #x_range=[0.4, 2.0],
                   kind='line', show=False, ax=axes[0])


  var1.fill_between(var1.lines[0].get_data()[0],
                    var1.lines[0].get_data()[1],
                    color='orange', alpha=0.3)
  if len(vars) == 2:
    var_str = f'{vars[0]} and {vars[1]}'
  else:
    var_str = f'{vars[0]}, {vars[1]} and {vars[2]}'
  var1.set_title(f'Variation of {var_str} with\n'\
                  f'increasing {x_descr}')
  var1.yaxis.tick_right()
  var1.axvspan(band_min, band_max, color='blue', alpha=0.2)

  # if there are 3 then plot the middle one too
  if len(vars) == 3:
    var =  plot_multi(df,
                      rename=dict(perturb_norm_bound='epsilon'),
                      # x_ids = 'epsilon',
                      x_ids=x_var,
                      var='',
                      legend=False,
                      value=ylabels[1],
                      include=[x_var, vars[1]],
                      order_by=x_var, ascending=True,
                      threshold=0.0,
                      tight=False,
                      # x_range=[0.4, 2.0],
                      kind='line', show=False, ax=axes[1])

    var.fill_between(var.lines[0].get_data()[0],
                      var.lines[0].get_data()[1],
                      color='orange', alpha=0.3)
    var.yaxis.tick_right()
    var.axvspan(band_min, band_max, color='blue', alpha=0.2)

  i = len(vars) - 1
  var2 = plot_multi(df,
                   rename=dict(perturb_norm_bound='epsilon'),
                   x_ids = x_var, var='',
                   value=ylabels[i],
                   legend=False,
                   include = [x_var, vars[i]],
                   order_by=x_var, ascending=True,
                   threshold=0.0,
                   tight=False,
                   #x_range=[0.4, 2.0],
                   kind='line', show=False, ax=axes[i])

  var2.yaxis.tick_right()
  var2.axvspan(band_min, band_max, color='blue', alpha=0.2)
  if params.get('hmax') and params.get('hmin'):
    var2.axhspan(params.hmin, params.hmax, color='green', alpha=0.2)
  var2.axvspan(band_min, band_max, color='blue', alpha=0.2)
  var2.fill_between(var2.lines[0].get_data()[0],
                    var2.lines[0].get_data()[1],
                    color='grey', alpha=0.3)
  #bar = seaborn.barplot(y='names', x='v1', data=d, ax=axes[1])
  var1.set(xlabel='', xticks=[])
  plt.rcParams["text.usetex"] = True
  var2.set_xlabel(xlabel)
  #auc.set_xlabel('$\\lambda$')
  if params.ymin2 and params.ymax2:
    var2.set_ylim(params.ymin2, params.ymax2)
  plt.show()


#dfw = pd.read_csv('~/Dropbox/Git/robulin/results/weights_236285.csv')
#plot_drop_arv(dfw, x_ids = 'features', prefix='browser_version')
#plot_drop_arv(dfw, x_ids = 'features')

#heat = seaborn.heatmap(d.set_index('names'), cbar=False, linewidths=0.1,
# ax=axes[0])

