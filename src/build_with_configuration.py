import pandas as pd

from helpers.cross_validate import cross_validate

from src import NN_LABELS_PATH, TRUE_LABELS_PATH, NN_INFORMATION_PATH, DATA_PATH

def run(network_metadata):
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