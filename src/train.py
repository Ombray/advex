from src.utils import grad, loss
import tensorflow as tf
tfe = tf.contrib.eager
from src.robust_logistic import RobustLogisticModel
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score
import numpy as np
import pandas as pd
from src.utils import Bunch, entropy
from sklearn.model_selection import train_test_split

# one step of training
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# global_step = tf.train.get_or_create_global_step()
#
# loss_value, grads = grad(model, inputs, labels, robust=True)
#
# print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
#                                           loss_value.numpy()))
#
# optimizer.apply_gradients(zip(grads, model.variables), global_step)
#
# print("Step: {},         Loss: {}".format(global_step.numpy(),
#                                           loss(model, inputs, labels).numpy()))



def train(model: RobustLogisticModel,
          dataset, robust=0.0, epochs=100, lr=0.01):
  # keep results for plotting
  train_loss_results = []
  train_accuracy_results = []
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
  optimizer = tf.train.AdamOptimizer(learning_rate=lr)
  #optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
  global_step = tf.train.get_or_create_global_step()

  for epoch in range(epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    #epoch_accuracy = tfe.metrics.Accuracy()

    # Training loop - using batches
    for x, y in dataset:
      # Optimize the model
      loss_value, grads = grad(model, x, y, robust=robust)
      optimizer.apply_gradients(zip(grads, model.variables),
                                global_step)
      # Track progress

      epoch_loss_avg(loss_value)  # add current batch loss
      # compare predicted label to actual label
      #epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    #train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
      print("Epoch {:03d}: Loss: {:.5f}".format(epoch,
                                                epoch_loss_avg.result()))

  return np.round(epoch_loss_avg.result().numpy(), 3)



def test(model: RobustLogisticModel, data, perturb = 0.0,
         attacker:RobustLogisticModel=None):
  if not attacker:
    attacker = model
  test_loss_avg = tfe.metrics.Mean()
  feat_attribs = []
  ytrue = ypred = np.array([])
  for (x, y) in data:
    # if perturb:
    #   x = model.perturb_continuous(x, y, robust=1.0)
    loss_value, y_ = loss(model, x, y, robust=perturb, attacker=attacker)
    ypred = np.append(ypred, y_.numpy().squeeze())
    ytrue = np.append(ytrue, y.numpy().squeeze())
    test_loss_avg(loss_value)
    feat_attrib_values = model.feature_attributions(x, y, perturb=perturb)
    feat_attribs += [feat_attrib_values]
  auc = roc_auc_score(ytrue, ypred)
  r2 = r2_score(ytrue, ypred)
  ypred_binary = 1 * (ypred > 0.5)
  acc = accuracy_score(ytrue, ypred_binary)
  feat_attribs = tf.concat(feat_attribs, axis = 0)
  feat_abs_attribs = tf.reduce_mean(tf.abs(feat_attribs), axis=0)
  ent = entropy(feat_abs_attribs)
  feat_avg_attribs = tf.reduce_mean(feat_attribs, axis=0)
  av_loss = test_loss_avg.result()
  print(f"Test set loss: {av_loss:.2}, AUC={auc:.2}, R2={r2:.2}, acc={acc:2}")
  weights = model.get()
  # print('Weights:')
  # print(weights)
  # print('Feature avg attribs: ')
  # print(feat_avg_attribs.numpy())
  # print('Feature abs attribs: ')
  # print(feat_abs_attribs.numpy())
  return Bunch(auc=np.round(auc,2),
               r2= np.round(r2,2),
               acc= np.round(acc,2),
               loss = np.round(av_loss.numpy(),3),
               coefs=np.round(weights['coefs'].squeeze(),2),
               bias=np.round(weights['bias'][0],2),
               attr_ave = np.round(feat_avg_attribs.numpy().squeeze(),2 ),
               attr_abs = np.round(feat_abs_attribs.numpy().squeeze(),2 ),
               ent = ent)


def run(config, X, y,
        robust_frac, epochs, batch_size=20, lr=0.01, verbose=False, seed=123):
  train_X, test_X, train_y, test_y = train_test_split(X, y)
  np.random.seed(seed)
  tf.set_random_seed(seed)

  def train_data():
    nrows = train_X.shape[0]
    return tf.data.Dataset. \
      from_tensor_slices((train_X, train_y)). \
      shuffle(nrows, seed=123).batch(batch_size)

  def test_data():
    nrows = test_X.shape[0]
    return tf.data.Dataset. \
      from_tensor_slices((test_X, test_y)). \
      shuffle(nrows, seed=123).batch(batch_size)


  model = RobustLogisticModel(1, config=config)
  rob = 0.0
  print(f'**** training on natural plus {rob} perturbed***')
  # custom train on natural data mixed with adversarially perturbed data
  train_loss = train(model, train_data(), robust=rob, epochs=epochs, lr=lr)
  nat_nat = test(model, test_data())
  nat_nat.update(train=rob, train_loss = train_loss, test=0)
  print(f'Perf on perturbed data:')
  nat_per = test(model, test_data(), perturb=1.0)
  nat_per.update(train=rob, train_loss = train_loss, test=1.0)

  rob = robust_frac
  np.random.seed(seed)
  tf.set_random_seed(seed)

  # model = RobustLogisticModel(1, config=config)
  print(f'**** training on natural plus {rob} perturbed***')
  # custom train on natural data mixed with adversarially perturbed data
  train_loss = train(model, train_data(), robust=rob, epochs=epochs, lr=lr)
  per_nat = test(model, test_data())
  per_nat.update(train=rob, train_loss=train_loss, test=0)
  print(f'Perf on {rob} perturbed data:')
  per_per = test(model, test_data(), perturb=rob)
  per_per.update(train=rob, train_loss=train_loss, test=rob)
  print(f'Perf on 100 pct perturbed data:')
  per_per_all = test(model, test_data(), perturb=1.0)
  per_per_all.update(train=rob, train_loss=train_loss, test=1.0)

  all_results = pd.DataFrame([
    nat_nat,
    nat_per,
    per_nat,
    per_per,
    per_per_all])

  pd.set_option('display.max_columns', 500)
  pd.set_option('max_colwidth', 1000)
  pd.set_option('display.width', 1000)

  all_results['l2'] = config.get('l2_epsilon')
  some_results = all_results[[
    'train', 'test', 'l2',
    'train_loss',
    'loss', 'auc', 'acc', 'r2',
    'bias', 'ent',
    # 'coefs'
    # 'attr_ave', 'attr_abs'
    ]]
  print(some_results)
  return all_results




