"""
Build NN using Keras functional API given specification and parameters
"""
from typing import List

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense

def build(structure: List[int], activation_function: str, loss='categorical_crossentropy', optimizer='adam'):
    # Assuming structure is none empty TODO handle case none empty structure
    """
    Args:
        structure: structure of network INCLUDING input and output layer
        activation_function:
        loss:
        optimizer:

    Returns:
        Compiled Keras model based on specification
    """

    # Input layer
    input_layer = Input(shape=(structure[0],))

    # Hidden Layers
    prev_layer = input_layer
    for x in range(1, len(structure) - 1):
        hidden_layer = Dense(structure[x], activation=activation_function)(prev_layer)
        prev_layer = hidden_layer

    # Output Layer
    output_layer = Dense(structure[len(structure) - 1], activation='softmax')(prev_layer)

    # Model input-hidden-output
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile Model
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


