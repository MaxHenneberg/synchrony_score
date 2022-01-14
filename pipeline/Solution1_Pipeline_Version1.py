from datetime import date, datetime

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

from autoencoders.AnomalyDetector import AnomalyDetector
from dataplayground import DataUtil
from dataplayground.DataUtil import normalizeData, slidingWindow
from utils.prepare_data import collectUserData

chunkSize = 256
autoencoder = AnomalyDetector(chunkSize)
user1, user2 = collectUserData("Eyeblink_Raw", lambda sheet: sheet["User1Blink"], lambda sheet: sheet["User2Blink"])


def processUser(userData, normalizer, windowSize, windowOffset):
    processedUser = list()
    for i in range(len(userData)):
        for j in slidingWindow(normalizer(userData[i]), windowSize, windowOffset):
            processedUser.append(j)
    return np.array(processedUser)


user1 = processUser(user1, normalizeData, 256, 256)
user2 = processUser(user2, normalizeData, 256, 256)


def filterBatchesWithLowEnergy(user1, user2):
    meanUser1 = np.mean(user1)
    meanUser2 = np.mean(user2)
    mean = (meanUser1 + meanUser2) / 2

    filteredUser1Batches = list()
    filteredUser2Batches = list()
    for i in range(len(user1)):
        batchMeanUser1 = np.mean(user1[i])
        batchMeanUser2 = np.mean(user2[i])
        if batchMeanUser1 > mean and batchMeanUser2 > mean:
            filteredUser1Batches.append(user1[i])
            filteredUser2Batches.append(user2[i])

    return filteredUser1Batches, filteredUser2Batches

print(f'Before Null {len(user1)}')

notNullUser1 = list()
notNullUser2 = list()
for i in range(len(user1)):
    user1Sum = sum(user1[i])
    user2Sum = sum(user2[i])
    if user1Sum != 0 and user2Sum != 0:
        notNullUser1.append(user1[i])
        notNullUser2.append(user2[i])

# notNullUser1, notNullUser2 = filterBatchesWithLowEnergy(user1, user2)

print(f'After Null {len(notNullUser2)}')

train_data1, test_data1 = train_test_split(np.array(notNullUser1), test_size=0.2, random_state=21)
train_data2, test_data2 = train_test_split(np.array(notNullUser2), test_size=0.2, random_state=21)

autoencoder.compile(optimizer='adam', loss='mae')
history = autoencoder.fit(train_data1, train_data2,
                          epochs=60,
                          validation_data=(test_data2, test_data2),
                          shuffle=True)

reconstruction = autoencoder.predict(test_data1)


# mae = []
# for i in range(len(reconstructions)):
#     user1Rec = np.squeeze(reconstructions[i][0])
#     user1 = np.squeeze(train_data[i][0])
#     user2Rec = np.squeeze(reconstructions[i][1])
#     user2 = np.squeeze(train_data[i][1])
#     mae1 = tf.keras.losses.mae(user1Rec, user1)
#     mae2 = tf.keras.losses.mae(user2Rec, user2)
#     mae.append((mae1 + mae2) / 2)

def calculateMAE(targetTimeSeries, reconstruction, originTimeSeries):
    result = list()
    dtype = [('mae', float), ('targetValue', object), ('reconstructedValue', object), ('originalValue', object)]
    for i in range(len(targetTimeSeries)):
        targetValue = targetTimeSeries[i]
        reconstructedValue = reconstruction[i]
        mae = tf.keras.losses.mae(reconstructedValue, targetValue).numpy()
        result.append((mae, targetValue, reconstructedValue, originTimeSeries[i]))
    return np.sort(np.array(result, dtype=dtype), order='mae')


maeList = calculateMAE(test_data2, reconstruction, test_data1)

lengthTestData = min(len(maeList), 10)
fig, axs = plt.subplots(lengthTestData * 3)
fig.set_size_inches(10, 70)
fig.set_dpi(200)
for i in range(lengthTestData):
    axs[3 * i].set_title('Test_data_1')
    axs[3 * i].set_ylim([0, 1])
    axs[3 * i].plot(np.arange(256), maeList[i]['originalValue'], 'b')
    axs[(3 * i) + 1].set_title('Test_data_2')
    axs[(3 * i) + 1].set_ylim([0, 1])
    axs[(3 * i) + 1].plot(np.arange(256), maeList[i]['targetValue'], 'y')
    axs[(3 * i) + 2].set_title(f'Reconstrcuted_Test_Data_2 | MAE: {round(maeList[i]["mae"], 4)}')
    axs[(3 * i) + 2].set_ylim([0, 1])
    axs[(3 * i) + 2].plot(np.arange(256), maeList[i]['reconstructedValue'], 'r')

plt.tight_layout()
DataUtil.savePlot(plt, 'Solution1', 'Version1')
