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


class AnomalyDetectorRawData(Model):
    def __init__(self, chunkSize, layerDefinitions: List[LayerDefinition]):
        super(AnomalyDetectorRawData, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(2, chunkSize, 1)),
                tf.keras.layers.Conv2D(layerDefinitions[0].filter, layerDefinitions[0].kernelSize,
                                       activation=layerDefinitions[0].activation, padding=layerDefinitions[0].padding,
                                       strides=layerDefinitions[0].strides),
                tf.keras.layers.Conv2D(layerDefinitions[1].filter, layerDefinitions[1].kernelSize,
                                       activation=layerDefinitions[1].activation, padding=layerDefinitions[1].padding,
                                       strides=layerDefinitions[1].strides),
                # tf.keras.layers.Reshape(target_shape=(-1,)),
                # tf.keras.layers.Dense(2, activation='relu')
            ])

        self.decoder = tf.keras.Sequential(
            [
                # tf.keras.layers.InputLayer(input_shape=2),
                # tf.keras.layers.Dense(units=256, activation='relu'),
                # tf.keras.layers.Reshape(target_shape=(2, 8, 16)),
                tf.keras.layers.Conv2DTranspose(layerDefinitions[2].filter, layerDefinitions[2].kernelSize,
                                                activation=layerDefinitions[2].activation,
                                                padding=layerDefinitions[2].padding,
                                                strides=layerDefinitions[2].strides),
                tf.keras.layers.Conv2DTranspose(layerDefinitions[3].filter, layerDefinitions[3].kernelSize,
                                                activation=layerDefinitions[3].activation,
                                                padding=layerDefinitions[3].padding,
                                                strides=layerDefinitions[3].strides),
                tf.keras.layers.Conv2D(1, kernel_size=(1, 1), activation='linear', padding='same')
            ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
