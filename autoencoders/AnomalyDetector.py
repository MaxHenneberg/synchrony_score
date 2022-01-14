import tensorflow as tf
from keras import Model
from keras.applications.densenet import layers


class AnomalyDetector(Model):
    def __init__(self, chunkSize):
        super(AnomalyDetector, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(chunkSize, activation="linear"),
            layers.Dense(32, activation="linear"),
            layers.Dense(16, activation="linear")]
        )

        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="linear"),
            layers.Dense(32, activation="linear"),
            layers.Dense(chunkSize, activation="linear")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded