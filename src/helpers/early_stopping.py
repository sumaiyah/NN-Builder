"""
Given X, y, evaluate model performance on a validation set every 10 epochs until performance drops
"""
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from helpers import build_network
from keras.callbacks import EarlyStopping, ModelCheckpoint


# https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
def early_stopping(X, y, network_info):
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

    # Weight classes due to imbalanced dataset
    class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                     np.unique(y_train), y_train)))

    # to get 2 node output convert target to one hot vector encoding
    y_train, y_val = to_categorical(y_train), to_categorical(y_val)

    model = build_network.build(structure=network_info.structure,
                                activation_function=network_info.activation_function)

    # Early stopping mechanism
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=100)

    # Train Model
    model.fit(X_train,
              y_train,
              validation_data=(X_val, y_val),
              epochs=4000, verbose=0,
              callbacks=[es],
              class_weight=class_weights)
