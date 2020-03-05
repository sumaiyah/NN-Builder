# Configurations for datasets
# TODO do this using proper configs later

from collections import namedtuple, OrderedDict

NetworkMetadata = namedtuple('NetworkMetadata', 'name target_col_name structure activation_function hyperparameters')

def get_configuration(dataset_name):
    """
    Return network metadata given the datasetname
    i.e. Class names with their corresponding encoding (output neuron index)
    """
    network_metadata = None

    if dataset_name == 'Artif-1':
        network_metadata = NetworkMetadata(name=dataset_name,
                                           target_col_name='y',
                                           structure=[5, 10, 5, 2],
                                           activation_function='tanh',
                                           hyperparameters=OrderedDict(batch_size=100, epochs=50, verbose=0))

    elif dataset_name == 'Artif-2':
        network_metadata = NetworkMetadata(name=dataset_name,
                                           target_col_name='y',
                                           structure=[5, 10, 5, 2],
                                           activation_function='tanh',
                                           hyperparameters=OrderedDict(batch_size=10, epochs=100, verbose=0))
    elif dataset_name == 'BreastCancer':
        network_metadata = NetworkMetadata(name=dataset_name,
                                           target_col_name='diagnosis',
                                           structure=[30, 50, 50, 2],
                                           activation_function='tanh',
                                           hyperparameters=OrderedDict(batch_size=20, epochs=100, verbose=0))
    elif dataset_name == 'MB-GE-ER':
        network_metadata = NetworkMetadata(name=dataset_name,
                                       target_col_name='ER_Expr',
                                       structure=[1000, 50, 5, 2],
                                       activation_function='tanh',
                                       hyperparameters=OrderedDict(batch_size=20, epochs=50, verbose=0))
    elif dataset_name == 'LetterRecognition-binary':
        network_metadata = NetworkMetadata(name=dataset_name,
                                       target_col_name='letter',
                                       structure=[16, 30, 10, 2],
                                       activation_function='tanh',
                                       hyperparameters=OrderedDict(batch_size=10, epochs=200, verbose=0))
    elif dataset_name == 'LetterRecognition-multi':
        network_metadata = NetworkMetadata(name=dataset_name,
                                       target_col_name='letter',
                                       structure=[16, 30, 10, 26],
                                       activation_function='tanh',
                                       hyperparameters=OrderedDict(batch_size=50, epochs=100, verbose=0))

    return network_metadata
