from collections import namedtuple
from random import randint

import numpy as np
from keras.utils import to_categorical
from sklearn.utils import class_weight

from model_building.cross_validate import cross_validate
from model_building import build_network

# ------------------------------------------ Initialise Parameters ----------------------------------------------------
import pandas as pd

from model_building.early_stopping import early_stopping

NetworkMetadata = namedtuple('NetworkMetadata', 'name structure activation_function batch_size epochs id')
network_info = NetworkMetadata(name='MB-ER',
                               activation_function='tanh',
                               structure=[1000, 100, 50, 2],
                               id=randint(0, 10000),
                               batch_size=15,
                               epochs=100)
datapath = 'preprocessing/%s/' % network_info.name

train_data = pd.read_csv(datapath + 'train_data.csv')
X_train = train_data.drop(['target'], axis=1).iloc[:, :].values
y_train = train_data['target'].values

test_data = pd.read_csv(datapath + 'test_data.csv')
X_test = test_data.drop(['target'], axis=1).iloc[:, :].values
y_test = test_data['target'].values

col_names = train_data.columns
# --------------------------------------------------------------------------------------------------------------------

# Only run final model training once with best parameter settings!

# ----------------------------------------- Cross Validation Step ----------------------------------------------------
# Cross Validate with different parameters until reach best accuracy on validation data
# cross_validate(X=X_train,
#                y=y_train,
#                n_folds=5,
#                network_info=network_info)
# --------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------- Find best number of epochs using early stopping ------------------
# early_stopping(X=X_train, y=y_train, network_info=network_info)
# --------------------------------------------------------------------------------------------------------------------


# ----------------------------------------- Final Model Training ---------------------------------------------------
# Use best parameter settings on final model Final model trained on training and validation data
model = build_network.build(structure=network_info.structure, activation_function=network_info.activation_function)

# Weight classes due to imbalanced dataset
class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)))

# Train Model
model.fit(X_train,
          to_categorical(y_train),
          batch_size=network_info.batch_size,
          epochs=network_info.epochs,
          verbose=0,
          class_weight=class_weights)

# Test Model
_, accuracy = model.evaluate(X_test, to_categorical(y_test))

# Save Model
filepath = 'output/'
model.save('%s%s_%d.h5' % (filepath, network_info.name, network_info.id))
# Store trained model information in .txt file
network_structure = '-'.join(str(layer) for layer in network_info.structure)
with open(filepath + 'information.txt', 'a') as file:
    file.write(str(network_info) + '\n')
    file.write('n_train: %d n_test: %d \n' % (len(X_train), len(X_test)))
    file.write('Model Accuracy: ' + str(accuracy) + '\n\n')
# --------------------------------------------------------------------------------------------------------------------

