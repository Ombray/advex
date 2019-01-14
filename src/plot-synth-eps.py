from src.plot import plot_multi
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#result_file = 'results/synth-results.csv'
#result_file = 'results/synth-eps-bad.csv'
result_file = 'results/synth-eps-bad20180926-141050.csv'
#result_file = 'results/synth-l1-results-corr.csv'
df = pd.read_csv(result_file)
fig, axes = plt.subplots(nrows=3, ncols=1, squeeze=True)
x_var = 'epsilon'
#x_var = 'l1_reg'
#fig.tight_layout(pad=0, h_pad=0, w_pad=0)

fig.subplots_adjust(hspace = 0.05, wspace=0.05)
band_min=0.6
band_max=1.2

wts_x = plot_multi(df,
                   rename=dict(perturb_norm_bound='epsilon',
                               Wts_x_L1='wts_x',
                               Wts_r_L1='wts_r'),
                   #x_ids = 'epsilon',
                   x_ids = x_var,
                   var='',
                   legend=False,
                   value='L1 norm of x wts',
                   include = [x_var, 'wts_x'],
                   order_by=x_var, ascending=True,
                   threshold=0.0,
                   #x_range=[0.4, 2.0],
                   kind='line', show=False, ax=axes[0])

wts_x.fill_between(wts_x.lines[0].get_data()[0],
                   wts_x.lines[0].get_data()[1],
                   color='grey', alpha=0.3)
wts_x.set_title('Variation of weights magnitude and AUC with \n'
                'increasing $\\ell_\infty$-bound $\\varepsilon$')
wts_x.yaxis.tick_right()
wts_x.axvspan(band_min, band_max, color='blue', alpha=0.2)
wts_r = plot_multi(df,
                   rename=dict(perturb_norm_bound='epsilon',
                               Wts_x_L1='wts_x',
                               Wts_r_L1='wts_r'),
                   x_ids = x_var, var='',
                   value='L1 norm of r wts',
                   legend=False,
                   include = [x_var, 'wts_r'],
                   order_by=x_var, ascending=True,
                   threshold=0.0,
                   #x_range=[0.4, 2.0],
                   kind='line', show=False, ax=axes[1])

wts_r.yaxis.tick_right()
wts_r.axvspan(band_min, band_max, color='blue', alpha=0.2)


wts_r.fill_between(wts_r.lines[0].get_data()[0],
                   wts_r.lines[0].get_data()[1],
                   color='orange', alpha=0.3)

#bar = seaborn.barplot(y='names', x='v1', data=d, ax=axes[1])

wts_x.set(xlabel='', xticks=[])
wts_r.set(xlabel='', xticks=[])

auc = plot_multi(df,
                 rename=dict(perturb_norm_bound='epsilon'),
                 x_ids = x_var, var='',
                 value='AUC',
                 include = [x_var, 'auc'],
                 legend=False,
                 order_by=x_var, ascending=True,
                 #x_range=[0.4, 2.0],
                 kind='line', show=False, ax=axes[2])

auc.yaxis.tick_right()
plt.rcParams["text.usetex"] =True
auc.set_xlabel('$\\epsilon$')
#auc.set_xlabel('$\\lambda$')
auc.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
auc.axvspan(band_min, band_max, color='blue', alpha=0.2)

plt.show()
