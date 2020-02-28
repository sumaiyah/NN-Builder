from collections import namedtuple

from helpers.configurations import NetworkMetadata, get_configuration
import build_with_configuration
import find_optimal_configuration

from src import DATASET_NAME

# Boolean flag.
# If True, build and train 5 neural networks with a given configuration
# If False, run grid search to find optimal hyper-parameters and then build the networks
given_configuration = True

if given_configuration:
    # Either define a configuration here or get a pre-defined configuration from configurations.py
    # Neural Network metadata
    # network_metadata = NetworkMetadata(name=DATASET_NAME,
    #                                            target_col_name='y',
    #                                            structure=[5, 10, 5, 2],
    #                                            activation_function='tanh',
    #                                            hyperparameters=OrderedDict(batch_size=500, epochs=100, verbose=0))
    network_metadata = get_configuration(DATASET_NAME)

else:
    # Find optimal network structure and hyper-parameters using a parameter sweep
    network_metadata = find_optimal_configuration.get_configuration(DATASET_NAME)


build_with_configuration.run(network_metadata=network_metadata)