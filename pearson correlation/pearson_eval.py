import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from timeit import default_timer as timer

import Utils
from dataplayground import DataUtil
from dataplayground.DataUtil import normalizeData, splitSeriesOverMultipleAxis
from utils.prepare_data import collectUserData


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


srcExcel = "User_Study_3MIN"
targetFolder = "PEARSON_EVAL"
user1GetColumn = lambda sheet: sheet["User1Smile"]
user2GetColumn = lambda sheet: sheet["User2Smile"]
user1PreProcessing = lambda data: data
user2PreProcessing = lambda data: data

r_window_size = 8

user1, user2 = collectUserData(srcExcel, user1GetColumn,
                               user2GetColumn)

# Normalize Time Series
user1 = [normalizeData(sheet) for sheet in user1]
user2 = [normalizeData(sheet) for sheet in user2]

# Apply Custom Preprocessing to normalized UserData
user1 = user1PreProcessing(user1)
user2 = user2PreProcessing(user2)

rollingRList = []

for study1, study2 in zip(user1, user2):
    dfUser1 = pd.DataFrame(list(study1), columns=['SyncScore']).interpolate()
    dfUser2 = pd.DataFrame(list(study2), columns=['SyncScore']).interpolate()

    start = timer()
    rolling_r = np.array(dfUser1['SyncScore'].rolling(window=r_window_size, center=True).corr(dfUser2['SyncScore']))
    rolling_r = np.nan_to_num(rolling_r)
    rolling_r = pow(rolling_r, 2)
    end = timer()
    rollingRList.append(rolling_r)

    s1 = dfUser1['SyncScore']
    s2 = dfUser2['SyncScore']
    fig, axs = plt.subplots(6, 1)
    fig.set_size_inches(14, 6)
    splitSeriesOverMultipleAxis(s1, axs, [0, 2, 4], Utils.TUM_BLUE)
    splitSeriesOverMultipleAxis(s2, axs, [0, 2, 4], Utils.TUM_ORANGE)
    splitSeriesOverMultipleAxis(rolling_r, axs, [1, 3, 5], Utils.TUM_GREEN)
    plt.show()

DataUtil.createDirIfNotExistent(targetFolder)
for i, mergedSyncScore in enumerate(rollingRList):
    DataUtil.saveDataThread(["SyncScore"], [mergedSyncScore], targetFolder,
                            f'Peasron_Eval_Study_{i}', '')

    # print(f'Took {end-start}')

