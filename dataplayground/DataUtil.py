import numpy
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import array


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


def slidingWindow(data: numpy.ndarray, windowSize: int, windowOffset: int):
    steps = (int)(data.size / windowOffset)
    resultArray = [data[0:windowSize]]
    for i in range(steps - 1):
        windowPos = ((i + 1) * windowOffset)
        if (windowPos + windowSize >= data.size):
            break
        resultArray = np.concatenate((resultArray, [data[windowPos:(windowPos + windowSize)]]))

    return resultArray
