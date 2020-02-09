"""
Given X, y perform cross validation
"""
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight

from helpers import build_network

def cross_validate(X, y, n_folds, network_info):
    skf = StratifiedKFold(n_splits=n_folds)

    # accuracies for each fold
    model_accuracies = []

    # train and test model over each fold
    for train_index, val_index in skf.split(X, y):
        model = build_network.build(structure=network_info.structure,
                                    activation_function=network_info.activation_function)

        X_train, X_val = X[train_index], X[val_index]

        # to get 2 node output convert target to one hot vector encoding
        y_train, y_val = to_categorical(y[train_index]), to_categorical(y[val_index])

        # Weight classes due to imbalanced dataset
        class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                         np.unique(y[train_index]), y[train_index])))

        # Train Model
        model.fit(X_train,
                  y_train,
                  batch_size=network_info.batch_size,
                  epochs=network_info.epochs,
                  verbose=0,
                  class_weight=class_weights)

        #   Test Model
        _, model_accuracy = model.evaluate(X_val, y_val)
        model_accuracies.append(model_accuracy)

        print('Fold index:' + str(len(model_accuracies)) + ' Accuracy: ' + str(model_accuracy))

    cross_validation_accuracy = sum(model_accuracies) / len(model_accuracies)
    print('\n Cross Validation Accuracy: ', cross_validation_accuracy)