"""
Given X, y perform cross validation
"""
import numpy as np
from keras.utils import to_categorical
import pandas as pd
from sklearn.utils import class_weight
from helpers import build_network
from helpers.get_fold_data import get_train_and_test_indices


def cross_validate(X, y, n_folds, network_metadata):
    from src.main import MODEL_PATH, FOLD_PATH, NN_INFORMATION_PATH, NN_LABELS_PATH, TRUE_LABELS_PATH
    """
    Args:
        X: input features
        y: target column
        n_folds: number of folds
        network_metadata: network metadata includes neural network hyper parameter values

    """
    # Accuracy of neural network model over each fold
    model_accuracies = []

    # Build, Train and Test Neural Network over each fold.
    for fold_index in range(0, n_folds):
        train_indices, test_indices = get_train_and_test_indices(fold_indices_path=FOLD_PATH, fold_index=fold_index)
        X_train, X_test = X[train_indices], X[test_indices]

        # To get 2 node output make y categorical
        y_train, y_test = to_categorical(y[train_indices]), to_categorical(y[test_indices])

        # Weight classes due to imbalanced dataset
        class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                         np.unique(y[train_indices]),
                                                                         y[train_indices])))

        # Build initial model
        model = build_network.build(structure=network_metadata.structure,
                                    activation_function=network_metadata.activation_function)

        # Train Model
        model.fit(X_train,
                  y_train,
                  class_weight=class_weights,
                  **network_metadata.hyperparameters)

        #   Test Model
        _, model_accuracy = model.evaluate(X_test, y_test)
        model_accuracies.append(model_accuracy)
        print('Fold %d Accuracy: %f' % (fold_index, model_accuracy))

        # Save model
        model.save(MODEL_PATH + 'fold_%d.h5' % fold_index)

        # Write Neural Network predictions to file
        network_predictions = np.argmax(model.predict(X_test), axis=1)
        with open(NN_LABELS_PATH, 'a') as file:
            file.write(' '.join([str(pred) for pred in network_predictions]))
            file.write('\n')

        # Write True labels to file
        true_classifications = np.argmax(y_test, axis=1)
        with open(TRUE_LABELS_PATH, 'a') as file:
            file.write(' '.join([str(pred) for pred in true_classifications]))
            file.write('\n')

        # Write Neural Network Metadata to information.txt
        with open(NN_INFORMATION_PATH, 'a') as file:
            file.write('fold %d accuracy: %f \n' % (fold_index, model_accuracy))

    # Compute Cross-validated accuracy as average accuracy and write to disk
    cv_accuracy = sum(model_accuracies) / len(model_accuracies)
    print('Cross-Val Accuracy: %f' % cv_accuracy)
    with open(NN_INFORMATION_PATH, 'a') as file:
        file.write('Cross-Val Accuracy: %f \n' % cv_accuracy)
        file.write('------------------------------------------------------------- \n')