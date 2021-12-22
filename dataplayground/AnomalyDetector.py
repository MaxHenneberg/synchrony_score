import tensorflow as tf
from keras import Model
from keras.applications.densenet import layers


class AnomalyDetector(Model):
    def __init__(self, chunkSize):
        super(AnomalyDetector, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu")]
        )

        self.decoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(chunkSize, activation="relu")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded