from timeit import default_timer as timer

import numpy as np
from dtw import accelerated_dtw

from dataplayground import DataUtil
from dataplayground.DataUtil import normalizeData
from utils.prepare_data import collectUserData

srcExcel = "User_Study_3MIN"
targetFolder = "DTW_EVAL"
user1GetColumn = lambda sheet: sheet["User1Smile"]
user2GetColumn = lambda sheet: sheet["User2Smile"]
user1PreProcessing = lambda data: data
user2PreProcessing = lambda data: data

user1, user2 = collectUserData(srcExcel, user1GetColumn,
                               user2GetColumn)

# Normalize Time Series
user1 = [normalizeData(sheet) for sheet in user1]
user2 = [normalizeData(sheet) for sheet in user2]

# Apply Custom Preprocessing to normalized UserData
user1 = user1PreProcessing(user1)
user2 = user2PreProcessing(user2)

syncScorePerStudy = []
for studyU1, studyU2 in zip(user1, user2):
    s1 = np.array(studyU1, dtype=np.double)
    s2 = np.array(studyU2, dtype=np.double)
    start = timer()
    d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(s1, s2, dist='euclidean')
    end = timer()
    print(f'Took {end-start} time')
    rawDiagonal = (1 - np.diagonal(cost_matrix))
    costDiagonal = [0 if (s1[i] == 0 or s2[i] == 0) else rawDiagonal[i] for i in range(len(rawDiagonal))]
    syncScorePerStudy.append(costDiagonal)

# fig, axs = plt.subplots(6, 1, constrained_layout=True)
# axs.flat[0].plot(np.arange(len(s1) // 2), s1[:len(s1) // 2], Utils.TUM_BLUE)
# axs.flat[1].plot(np.arange(len(s1) // 2), s2[:len(s1) // 2], Utils.TUM_BLUE)
# axs.flat[2].plot(np.arange(len(s1) // 2), costDiagonal[:len(s1) // 2], Utils.TUM_GREEN)
# axs.flat[3].plot(np.arange(len(s1) // 2), s1[len(s1) // 2:], Utils.TUM_BLUE)
# axs.flat[4].plot(np.arange(len(s1) // 2), s2[len(s1) // 2:], Utils.TUM_BLUE)
# axs.flat[5].plot(np.arange(len(s1) // 2), costDiagonal[len(s1) // 2:], Utils.TUM_GREEN)
# plt.show()

DataUtil.createDirIfNotExistent(targetFolder)
for i, mergedSyncScore in enumerate(syncScorePerStudy):
    DataUtil.saveDataThread(["SyncScore"], [mergedSyncScore], targetFolder,
                        f'DTW_Eval_Study_{i}', '')
