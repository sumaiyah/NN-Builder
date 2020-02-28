from src import DATA_PATH, FOLD_PATH

def get_configuration(dataset_name):
    """

    Args:
        dataset_name:

    Returns:
    Optimal configuration for a neural network trained on this dataset
    """

    # Define ranges of parameters to grid search over and find optimal hyper-parameters
    # Train neural networks to early stopping?
