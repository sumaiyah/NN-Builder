def get_train_and_test_indices(fold_indices_path, fold_index):
    """
    Return train and test indices for a given fold

    File of the form
    fold0
    train 0 1 0 2 ...
    test 3 4 6 ...
    fold1
    train 1 5 2 ...
    test 6 8 9 ...
    ...
    """
    with open(fold_indices_path, 'r') as file:
        lines = file.readlines()
        assert len(lines) >= (3 * fold_index) + 2, 'Error: not enough information in fold indices file'

        train_indices = lines[(fold_index * 3) + 1].split(' ')[1:]
        test_indices = lines[(fold_index * 3) + 2].split(' ')[1:]

        # Convert string indicies to ints
        train_indices = [int(i) for i in train_indices]
        test_indices  = [int(i) for i in test_indices]

    return train_indices, test_indices
