from typing import List

import tensorflow as tf
from keras import Model
from keras.applications.densenet import layers


class LayerDefinition:
    filter = None
    kernelSize = None
    strides = None
    activation = None
    padding = None

    def __init__(self, filter, kernelSize, activation, padding, strides):
        self.filter = filter
        self.kernelSize = kernelSize
        self.strides = strides
        self.activation = activation
        self.padding = padding

    def toString(self):
        template = '#Filter: {filter}, Kernel-Size:{kernelSize}, Activation: {activation}, Padding: {padding}, Strides: {strides}'
        return (template.format(filter=self.filter, kernelSize=self.kernelSize, activation=self.activation,
                                padding=self.padding, strides=self.strides))


class AnomalyDetectorU1ToU2(Model):
    def __init__(self, chunkSize, layerDefinitions: List[LayerDefinition]):
        super(AnomalyDetectorU1ToU2, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(256, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(8, activation="relu")]
        )

        self.decoder = tf.keras.Sequential([
            layers.Dense(8, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(chunkSize, activation="relu")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
