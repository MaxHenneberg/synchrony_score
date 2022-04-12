import numpy as np
from matplotlib import pyplot as plt

from dataplayground import DataUtil
from utils.prepare_data import load_session_data_general

syncScore = load_session_data_general("..\\resources\\User_Study", "UserStudy_Summary", ["SyncScore"])["SyncScore"]

mergedSyncScorePerVideo = [np.array([]) for i in range(3)]
countPerVideo = [0 for i in range(3)]
for (study, syncScoreEntry) in zip(syncScore["study"], syncScore["syncScore"]):
    countPerVideo[study] = countPerVideo[study] + 1
    if (len(mergedSyncScorePerVideo[study]) == 0):
        mergedSyncScorePerVideo[study] = np.array(syncScoreEntry.split(), dtype=float)
    else:
        mergedSyncScorePerVideo[study] = mergedSyncScorePerVideo[study] + np.array(syncScoreEntry.split(), dtype=float)

for i, count in enumerate(countPerVideo):
    mergedSyncScorePerVideo[i] = mergedSyncScorePerVideo[i] / count

print(mergedSyncScorePerVideo[0])

# fig, axs = plt.subplots(3, 1, constrained_layout=True)
# for i, mergedSyncScore in enumerate(mergedSyncScorePerVideo):
#     axs.flat[i].plot(np.arange(len(mergedSyncScore)), mergedSyncScore)
#     axs.flat[i].plot(np.arange(len(mergedSyncScore)), mergedSyncScore)
#     axs.flat[i].plot(np.arange(len(mergedSyncScore)), mergedSyncScore)
# plt.show()

DataUtil.createDirIfNotExistent("User_Study_Eval")
for i, mergedSyncScore in enumerate(mergedSyncScorePerVideo):
    DataUtil.saveDataThread(["SyncScore"], [mergedSyncScore], "User_Study_Eval",
                        f'User_Study_Eval_Study_{i}', '')
