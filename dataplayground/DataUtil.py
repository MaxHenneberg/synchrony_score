import _thread
import os
from pathlib import Path
from typing import List

import numpy
import numpy as np
import pandas as pd
import tensorflow as tf

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
    if min_val == max_val:
        if min_val > 0:
            return data // min_val
        else:
            return data

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
        partSum = prevSize / currentLayer.strides[0]
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


WEIGHTS_STORE_PATH = './weights/'


def saveWeights(model, name):
    model.save_weights(WEIGHTS_STORE_PATH + name)


def loadWeights(model, name):
    model.load_weights(WEIGHTS_STORE_PATH + name)


plotPath = '..\\results\\plots\\'
dataPath = '..\\results\\data\\'


def createDirIfNotExistent(folder):
    Path(os.path.join(dataPath, folder)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(plotPath, folder)).mkdir(parents=True, exist_ok=True)


def saveFigure(fig, folder, name, runId):
    name = os.path.join(folder, name + '-' + runId)
    fig.savefig(os.path.join(plotPath, name))
    print(f'Plot stored to {name}.png')


def saveData(columnNames, data, folder, name, runId):
    _thread.start_new_thread(saveDataThread, (columnNames, data, folder, name, runId))


def saveDataThread(columnNames, data, folder, name, runId):
    path = os.path.join(os.path.join(dataPath, folder), f"{name}-{runId}.xlsx")
    xls_writer = pd.ExcelWriter(
        path,
        engine="xlsxwriter")
    for i, sheet in enumerate(data):
        sheetNumber = calcSheetNumber(i)
        df = pd.DataFrame(data=np.array(sheet).transpose(), columns=columnNames)
        df.to_excel(xls_writer, sheet_name=f"Study{sheetNumber}")
    xls_writer.save()
    print(f'Data stored to {path}')


def calcSheetNumber(s):
    sheetNumber = s + 1
    # There is no study 4
    if sheetNumber > 3:
        sheetNumber = sheetNumber + 1
    return sheetNumber


def splitSeriesOverMultipleAxis(data, axs, listOfAxsIdxs, color, labelStr=None, yLim=None):
    splits = np.split(np.array(data), len(listOfAxsIdxs))
    prevLen = 0
    for i, (idx, split) in enumerate(zip(listOfAxsIdxs, splits)):
        minVal = min(split)
        maxVal = max(split)
        if yLim == None:
            axs.flat[idx].set_ylim([minVal - max(0.1, abs(minVal * 0.05)), maxVal + max(0.1, abs(maxVal * 0.05))])
        else:
            axs.flat[idx].set_ylim(yLim)
        if i == 0:
            axs.flat[idx].plot(np.arange(len(split)) + prevLen, split, color, label=labelStr)
        else:
            axs.flat[idx].plot(np.arange(len(split)) + prevLen, split, color)
        prevLen = prevLen + len(split)