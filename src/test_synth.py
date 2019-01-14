import tensorflow as tf
from src.robust_logistic import RobustLogisticModel
from data_loader.synthetic_data_generator import SyntheticDataGenerator
from src.train import run
from src.utils import loss, grad, Bunch
tf.enable_eager_execution()
import numpy as np
import pandas as pd


# evaluation on natural test data

config = Bunch(N=1000, batch_size=20,
               perturb_one_hot=False,
               col_spec=[
                 dict(type='cat', card=3, name = 'c1'),
                 dict(type='cat', card=4, name = 'c2'),
                 dict(type='num', min=0, max=6, name='n1'),
                 dict(type='num', min=-3, max=3, name='n2')],
               noise=0.5,
               coefs=np.array([-1.0, 1.0, 0.4,  # cat feature 1
                 -1.0, 3.0, -1.0, -0.5,  # cat feature 2
                 0.00001,  # n1
                 0.7]),
               bias = 0.5
               )

train_data = SyntheticDataGenerator(config).data
test_data = SyntheticDataGenerator(config).data

l2 = 3
model = RobustLogisticModel(1, l2_epsilon=l2, config=config)


inputs, y = next(iter(train_data()))

y_ = model(inputs) # just to build the model and get input shapes

# run an exp
run(config, train_data, test_data, l2, robust_frac=0.5, epochs=4, lr=0.01)

# test various perturbations
inputs_pert_num = model.perturb_numeric(inputs, y)

inputs_pert_cat = model.perturb_categorical(inputs, y)



a = 1



