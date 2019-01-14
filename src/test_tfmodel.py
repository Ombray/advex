import tensorflow as tf
from src.robust_logistic import RobustLogisticModel
from data_loader.synthetic_data_generator import SyntheticDataGenerator
from src.train import train, test
from src.utils import loss, grad, Bunch
tf.enable_eager_execution()
import numpy as np
import pandas as pd

config = Bunch(N=500, batch_size=5,
               col_spec=[
                 #dict(type='cat', card=3),
                 #dict(type='cat', card=5),
                 dict(type='num', min=0, max=6, name='n1'),
                 dict(type='num', min=-3, max=3, name='n2')],
               noise=0.4,
               coefs=np.array([#-1.0, 1.0, 0.4,  # cat feature 1
                               #-1.0, 3.0, 0.9, -1.0, -0.5,  # cat feature 2
                               0.8,  # n1
                               -0.9]),
               bias = 0.3
               )
syndata = SyntheticDataGenerator(config)

# model = RobustLogisticModel(1, l2_epsilon=0.1,  config=config)
# wts = model.get_weights()
#
# # testing
# inputs, labels = syndata.next_batch()
# out = model(inputs, label=1)
# attribs = model.attributions(inputs, labels)
# feat_attribs = model.feature_attributions(inputs, labels)
# model_loss = loss(model, inputs, labels)
# grads = grad(model, inputs, labels, robust=0.5)


# compile and train using standard API
# Doesn't work
# mdl = tf.keras.Sequential([model])
# mdl.compile(optimizer='GradientDescent',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test)

# evaluation on natural test data

config = Bunch(N=400, batch_size=20,
               col_spec=[
                 #dict(type='cat', card=3, name = 'c1'),
                 #dict(type='cat', card=5),
                 dict(type='num', min=0, max=6, name='n1'),
                 dict(type='num', min=-3, max=3, name='n2')],
               noise=2.0,
               coefs=np.array([#-1.0, 1.0, 0.4,  # cat feature 1
                 #-1.0, 3.0, 0.9, -1.0, -0.5,  # cat feature 2
                 0.00001,  # n1
                 0.7]),
               bias = 0.5
               )

l2 = 0.5
model = RobustLogisticModel(1, l2_epsilon=l2, config=config)



syndata = SyntheticDataGenerator(config)
test_data = SyntheticDataGenerator(config)

rob = 0.0
print(f'**** training on natural plus {rob} perturbed***')
# custom train on natural data mixed with adversarially perturbed data
train(model, syndata.data(), robust=rob, epochs=40)
nat_nat = test(model, test_data.data())
nat_nat.update(train='nat', test='nat')
print(f'Perf on perturbed data:')
nat_per = test(model, test_data.data(), perturb=True)
nat_per.update(train='nat', test='per')

rob = 1.0
print(f'**** training on natural plus {rob} perturbed***')
# custom train on natural data mixed with adversarially perturbed data
train(model, syndata.data(), robust=rob, epochs=20)
per_nat = test(model, test_data.data())
per_nat.update(train='per', test='nat')
print(f'Perf on perturbed data:')
per_per = test(model, test_data.data(), perturb=True)
per_per.update(train='per', test='per')

# continue training on perturbed data
# print('**** adv training ***')
# train(model, syndata.data, robust=True, epochs=3)
# test(model, test_data)

print('**** true model test ***')
model_true = model
model_true.set(config.coefs, bias=config.bias)
tru_nat = test(model_true, test_data.data())
tru_nat.update(train='tru', test='nat')
print(f'Perf on perturbed data:')
tru_per = test(model_true, test_data.data(), perturb=True)
tru_per.update(train='tru', test='per')


all_results = pd.DataFrame([
  nat_nat,
  nat_per,
  per_nat,
  per_per,
  tru_nat,
  tru_per])

pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 1000)
pd.set_option('display.width', 1000)

all_results['l2'] = l2
all_results['noise'] = config.noise
print(all_results[['train', 'test', 'l2', 'noise',
                   'loss', 'auc', 'acc', 'r2', 'coefs', 'bias',
                   'attr_ave', 'attr_abs']])




