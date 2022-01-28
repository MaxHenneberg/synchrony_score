import os

import numpy as np
from matplotlib import pyplot as plt

from dataplayground.DataUtil import normalizeData
from utils.prepare_data import collectUserData, collectExcelData

windowSize = 8
wordSize = 8
batchSize = 256
folder = 'BOW'
user1, user2 = collectUserData("Eyeblink_Raw", lambda sheet: sheet["User1Blink"], lambda sheet: sheet["User2Blink"])
user1 = [normalizeData(sheet) for sheet in user1]
user2 = [normalizeData(sheet) for sheet in user2]

baseResultPath = "..\\results\\data"


syncScoreBlink = collectExcelData(os.path.join(baseResultPath, "BOW_Continuous_Merged"), "SyncScore-28_01_2022_09_57_18", lambda sheet: sheet["syncScore"])
syncScoreSmile = collectExcelData(os.path.join(baseResultPath, "BOW_Continuous_Merged_Smile"), "SyncScore-2Bins-28_01_2022_13_00_53", lambda sheet: sheet["syncScore"])
syncScores = [syncScoreBlink, syncScoreSmile]
scoreWeights = [1, 1]
cntSyncScores = len(syncScores)

mergedSyncScores = []
for i, syncScore in enumerate(syncScores):
    for j, sheet in enumerate(syncScore):
        if i == 0:
            mergedSyncScores.append(np.array(sheet) * scoreWeights[i])
        else:
            mergedSyncScores[j] = mergedSyncScores[j] + (np.array(sheet) * scoreWeights[i])

for i in range(len(mergedSyncScores)):
    mergedSyncScores[i] = mergedSyncScores[i] / cntSyncScores


fig, axs = plt.subplots(len(mergedSyncScores))
fig.set_size_inches(70, 20)
for ax, data in zip(axs, mergedSyncScores):
    ax.set_ylim([-0.1, 1])
    ax.plot(np.arange(len(data)), data)

plt.show()
