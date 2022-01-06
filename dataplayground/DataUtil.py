from typing import List

import numpy
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import array

from dataplayground.AnomalyDetectorRawData import LayerDefinition

def extractUser(filePath):
    ff_raw = pd.read_excel(filePath)
    ff = ff_raw.to_numpy()

    ff_user_1 = ff[:, 3]
    ff_user_2 = ff[:, 4]
    ff_frame = ff[:, 1]
    return ff_user_1, ff_user_2, ff_frame


def splitIntoTrainingChunks(array, size):
    splits = (int)(array.size / size)
    return np.row_stack(np.array_split(array[:(size * splits)], splits))


def calcSync(user1, user2):
    ff_diff = np.abs(user1 - user2)
    ff_sum = user1 + user2

    return ff_sum - ff_diff


def normalizeData(data):
    min_val = tf.reduce_min(data)
    max_val = tf.reduce_max(data)

    data_norm = (data - min_val) / (max_val - min_val)

    return data_norm


def normalizeData(data):
    min_val = tf.reduce_min(data)
    max_val = tf.reduce_max(data)

    data_norm = (data - min_val) / (max_val - min_val)

    return data_norm


def normalizeDataMinusOneOne(data):
    min_val = tf.reduce_min(data)
    max_val = tf.reduce_max(data)

    data_norm = 2 * ((data - min_val) / (max_val - min_val)) - 1

    return data_norm


def slidingWindowMultipleArrays(data: numpy.ndarray, windowSize: int, windowOffset: int):
    dataWithWindow = slidingWindow(data[0], windowSize, windowOffset)
    for i in range(len(data) - 1):
        dataWithWindow = np.concatenate((dataWithWindow, slidingWindow(data[i + 1], windowSize, windowOffset)))

    return dataWithWindow


def slidingWindow(data: numpy.ndarray, windowSize: int, windowOffset: int):
    steps = (int)(len(data) / windowOffset)
    resultArray = [data[0:windowSize]]
    for i in range(steps - 1):
        windowPos = ((i + 1) * windowOffset)
        if (windowPos + windowSize >= len(data)):
            break
        resultArray = np.concatenate((resultArray, [data[windowPos:(windowPos + windowSize)]]))

    return resultArray


def findInterestingChunks(user1: numpy.ndarray, user2: numpy.ndarray, chunkSize: int):
    user1FilterArray = []
    user2FilterArray = []
    i = 0
    while i < (user1.size - chunkSize):
        if (user1[i] > 0) | (user2[i] > 0):
            if (len(user1FilterArray) == 0):
                user1FilterArray = [user1[i:(i + chunkSize)]]
                user2FilterArray = [user2[i:(i + chunkSize)]]
            else:
                user1FilterArray = np.concatenate((user1FilterArray, [user1[i:(i + chunkSize)]]))
                user2FilterArray = np.concatenate((user2FilterArray, [user2[i:(i + chunkSize)]]))
        i = i + chunkSize

    return user1FilterArray, user2FilterArray

def calcLatentLength(chunkSize, layerDefinitions: List[LayerDefinition]):
    prevSize = chunkSize
    for i in range((int)((len(layerDefinitions) / 2))):
        currentLayer = layerDefinitions[i]
        partSum = prevSize / currentLayer.strides[1]
        if (round(partSum, 0) < partSum):
            prevSize = round(prevSize + 1, 0)
        else:
            prevSize = partSum
    return prevSize

def shapeInput(user1, user2, expandAxis):
    output = [np.expand_dims((user1[0], user2[0]), axis=expandAxis)]
    for i in range(len(user1) - 1):
        outStep = [np.expand_dims((user1[i + 1], user2[i + 1]), axis=expandAxis)]
        output = np.concatenate((output, outStep))
    return np.array(output)

# def shapeInput(user1, user2):
#     output = [(user1[0], user2[0])]
#     for i in range(len(user1) - 1):
#         outStep = [(user1[i + 1], user2[i + 1])]
#         output = np.concatenate((output, outStep))
#     return np.array(output)


def printStats(chunkSize, epochs, history, normal, layerDefinitions: List[LayerDefinition]):
    template1 = "AutoEncoder Config:\n -{cntLayers} Conv Layers:\n"
    layerTemplate = "\t- {}\n"
    template3 = "- chunkSize: {chunkSize}\n- Epochs: {epochs}\n- Result: {normal}\n- train_loss: {train_loss:.4f}; val_loss: {val_loss:.4f}\n- Latent-Layer Size: {latentLayer}"
    resultString = template1.format(cntLayers=len(layerDefinitions))

    latentLayerLength = calcLatentLength(chunkSize, layerDefinitions)
    latentLayerSize = latentLayerLength * layerDefinitions[1].filter

    for i in range(len(layerDefinitions)):
        resultString += layerTemplate.format(layerDefinitions[i].toString())

    resultString += template3.format(chunkSize=chunkSize, epochs=epochs,
                                     normal=', '.join(map(lambda x: 'T' if x else 'F', normal)),
                                     train_loss=history.history['loss'][epochs - 1],
                                     val_loss=history.history['val_loss'][epochs - 1], latentLayer=latentLayerSize)
    print(resultString)

WEIGHTS_STORE_PATH='./weights/'

def saveWeights(model, name):
    model.save_weights(WEIGHTS_STORE_PATH+name)

def loadWeights(model, name):
    model.load_weights(WEIGHTS_STORE_PATH+name)
