import numpy
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn as sk
from keras import Model
from keras.applications.densenet import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from dataplayground.AnomalyDetector import AnomalyDetector
from dataplayground.DataUtil import *

chunkSize = 250
autoencoder = AnomalyDetector(chunkSize)

ff_user_1, ff_user_2, ff_frame = extractUser('../resources/FF.xlsx')
ff_sync = calcSync(ff_user_1, ff_user_2)


ff_sync_splits = splitIntoTrainingChunks(ff_sync, chunkSize)

train_data, test_data = train_test_split(ff_sync_splits, test_size=0.2, random_state=21)
train_data = normalizeData(train_data)
test_data = normalizeData(test_data)

autoencoder.compile(optimizer='adam', loss='mae')
history = autoencoder.fit(train_data, train_data,
                          epochs=60,
                          validation_data=(test_data, test_data),
                          shuffle=True)

# plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.legend()
# plt.show()

# Show encoded/decoded data compared to original

# encoded_data = autoencoder.encoder(test_data).numpy()
# decoded_data = autoencoder.decoder(encoded_data).numpy()
#
# plt.plot(test_data[5], 'b')
# plt.plot(decoded_data[5], 'r')
# plt.fill_between(np.arange(300), decoded_data[5], test_data[5], color='lightcoral')
# plt.legend(labels=["Input", "Reconstruction", "Error"])
# plt.show()

# Plot Mean Average Error

reconstructions = autoencoder.predict(train_data)
train_loss = tf.keras.losses.mae(reconstructions, train_data)
#
# plt.hist(train_loss[None,:], bins=50)
# plt.xlabel("Train loss")
# plt.ylabel("No of examples")
# plt.show()

threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)


def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)


normal = predict(autoencoder, test_data, threshold)

test_data_size = (int)(tf.size(test_data) / chunkSize)
fig, axs = plt.subplots(test_data_size)
fig.suptitle('Test')
for i in range(test_data_size):
    axs[i].plot(np.arange(chunkSize), test_data[i], 'g' if normal[i] else 'r')

plt.show()
