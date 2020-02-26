import numpy as np
import pandas as pd
from collections import OrderedDict, namedtuple

from helpers.cross_validate import cross_validate
from helpers.configurations import get_configuration

# Neural Network metadata
NetworkMetadata = namedtuple('NetworkMetadata', 'name target_col_name structure activation_function hyperparameters')
network_metadata = NetworkMetadata(name='Artif-1',
                                           target_col_name='y',
                                           structure=[5, 10, 5, 2],
                                           activation_function='tanh',
                                           hyperparameters=OrderedDict(batch_size=500, epochs=100, verbose=0))
# network_metadata = get_configuration('MB-ER')

# Configure file paths and initialise files
BASE_PATH = 'C:/Users/sumaiyah/OneDrive - University Of Cambridge/Project/data/%s/' % network_metadata.name
MODEL_PATH = BASE_PATH + 'models/'
LABEL_PATH = BASE_PATH + 'labels/'

DATA_PATH = BASE_PATH + 'data.csv'
FOLD_PATH = BASE_PATH + 'fold_indices.txt'

NN_INFORMATION_PATH = BASE_PATH + 'information.txt'

NN_LABELS_PATH = LABEL_PATH + 'NN_labels.txt'
TRUE_LABELS_PATH = LABEL_PATH + 'TRUE_labels.txt'

if __name__ == "__main__":
    # Initialise txt files - clear old contents
    open(NN_LABELS_PATH, 'w').close()
    open(TRUE_LABELS_PATH, 'w').close()
    open(NN_INFORMATION_PATH, 'w').close()

    # Write Neural Network Metadata to information.txt
    with open(NN_INFORMATION_PATH, 'a') as file:
        file.write('Neural Network: \n')
        file.write(str(network_metadata) + '\n')

    # Load all data
    target_col = network_metadata.target_col_name
    data = pd.read_csv(DATA_PATH)
    X = data.drop([target_col], axis=1).values
    y = data[target_col].values

    # Split data into 5 folds and build NN that classifies data for each fold
    # Saves each neural network to disk
    # Saves all neural network predictions to disk
    # Saves all true classifications to disk
    cross_validate(X=X, y=y, n_folds=5, network_metadata=network_metadata)
