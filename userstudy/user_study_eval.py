import numpy as np
from matplotlib import pyplot as plt

import Utils
from dataplayground import DataUtil
from dataplayground.DataUtil import splitSeriesOverMultipleAxis, normalizeData
from utils.prepare_data import load_session_data_general, collectUserData

srcExcel = "User_Study_3MIN"
user1GetColumn = lambda sheet: sheet["User1Smile"]
user2GetColumn = lambda sheet: sheet["User2Smile"]
user1PreProcessing = lambda data: data
user2PreProcessing = lambda data: data

r_window_size = 8

user1List, user2List = collectUserData(srcExcel, user1GetColumn,
                               user2GetColumn)

# Normalize Time Series
user1List = [normalizeData(sheet) for sheet in user1List]
user2List = [normalizeData(sheet) for sheet in user2List]

# Apply Custom Preprocessing to normalized UserData
user1List = user1PreProcessing(user1List)
user2List = user2PreProcessing(user2List)

syncScoreUserStudy0 = list(
    load_session_data_general("..\\results\\data\\User_Study_Eval", "User_Study_Eval_Study_0-", ["Study1"])["Study1"][
        "SyncScore"])
syncScoreUserStudy1 = list(
    load_session_data_general("..\\results\\data\\User_Study_Eval", "User_Study_Eval_Study_1-", ["Study1"])["Study1"][
        "SyncScore"])
syncScoreUserStudy2 = list(
    load_session_data_general("..\\results\\data\\User_Study_Eval", "User_Study_Eval_Study_2-", ["Study1"])["Study1"][
        "SyncScore"])

syncScoreBow = load_session_data_general("..\\results\\data\\User_Study_3MIN",
                                         "SyncScore-User_Study_3MIN-(8, 4, 3, 1)-08_04_2022_14_16_00",
                                         ["Study1", "Study2", "Study3"])

syncScoreDTW0 = list(
    load_session_data_general("..\\results\\data\\DTW_EVAL", "DTW_Eval_Study_0-", ["Study1"])["Study1"][
        "SyncScore"])
syncScoreDTW1 = list(
    load_session_data_general("..\\results\\data\\DTW_EVAL", "DTW_Eval_Study_1-", ["Study1"])["Study1"][
        "SyncScore"])
syncScoreDTW2 = list(
    load_session_data_general("..\\results\\data\\DTW_EVAL", "DTW_Eval_Study_2-", ["Study1"])["Study1"][
        "SyncScore"])

syncScorePearson0 = list(
    load_session_data_general("..\\results\\data\\PEARSON_EVAL", "Peasron_Eval_Study_0-", ["Study1"])["Study1"][
        "SyncScore"])
syncScorePearson1 = list(
    load_session_data_general("..\\results\\data\\PEARSON_EVAL", "Peasron_Eval_Study_1-", ["Study1"])["Study1"][
        "SyncScore"])
syncScorePearson2 = list(
    load_session_data_general("..\\results\\data\\PEARSON_EVAL", "Peasron_Eval_Study_2-", ["Study1"])["Study1"][
        "SyncScore"])

syncScoreBowStudy0 = list(syncScoreBow["Study1"]["syncScore"])
syncScoreBowStudy1 = list(syncScoreBow["Study2"]["syncScore"])
syncScoreBowStudy2 = list(syncScoreBow["Study3"]["syncScore"])

syncScoreUserStudyList = [syncScoreUserStudy0, syncScoreUserStudy1, syncScoreUserStudy2]
syncScoreBowList = [syncScoreBowStudy0, syncScoreBowStudy1, syncScoreBowStudy2]
syncScoreDTWList = [syncScoreDTW0, syncScoreDTW1, syncScoreDTW2]
syncScorePearsonList = [syncScorePearson0, syncScorePearson1, syncScorePearson2]

for i, (userStudy, bow, dtw, pearson, user1, user2) in enumerate(
        zip(syncScoreUserStudyList, syncScoreBowList, syncScoreDTWList, syncScorePearsonList, user1List, user2List)):
    # fig, axs = plt.subplots(6, 1, constrained_layout=True)
    # fig.set_size_inches(8, 12)
    # fig.suptitle(f'Dyad {i + 1}')
    # axs.flat[0].set_title('Bow')
    # axs.flat[0].plot(np.arange(len(bow) // 2), bow[:len(bow) // 2], Utils.TUM_GREEN)
    # axs.flat[1].plot(np.arange(len(bow) // 2) + len(bow) // 2, bow[len(bow) // 2:], Utils.TUM_GREEN)
    # axs.flat[2].set_title('User Study')
    # axs.flat[2].plot(np.arange(len(userStudy) // 2), userStudy[:len(userStudy) // 2], Utils.TUM_BLUE)
    # axs.flat[3].plot(np.arange(len(userStudy) // 2) + (len(userStudy) // 2), userStudy[len(userStudy) // 2:], Utils.TUM_BLUE)
    # axs.flat[4].set_title('Both')
    # axs.flat[4].plot(np.arange(len(bow) // 2), bow[:len(bow) // 2], Utils.TUM_GREEN)
    # axs.flat[5].plot(np.arange(len(bow) // 2) + (len(bow) // 2), bow[len(bow) // 2:], Utils.TUM_GREEN)
    # axs.flat[4].plot(np.arange(len(userStudy) // 2), userStudy[:len(userStudy) // 2], Utils.TUM_BLUE)
    # axs.flat[5].plot(np.arange(len(userStudy) // 2) + (len(userStudy) // 2), userStudy[len(userStudy) // 2:], Utils.TUM_BLUE)
    # DataUtil.saveFigure(fig, 'User_Study_Eval', f'Dyad {i+1}','')
    # fig.clear()


    fig, axs = plt.subplots(10, 1, figsize=(10,14), constrained_layout=True)
    # fig.set_size_inches(14, 10)
    # plt.tight_layout(h_pad=0.05)
    # plt.set
    # plt.subplots_adjust(hspace=0.01)

    # fig.suptitle(f'Dyad {i + 1}')
    splitSeriesOverMultipleAxis(user1, axs, [0, 3, 6], Utils.TUM_ORANGE, labelStr='User1', yLim=[-0.01, 1.1])
    splitSeriesOverMultipleAxis(user2, axs, [0, 3, 6], Utils.TUM_DARK_GRAY, labelStr='User2', yLim=[-0.01, 1.1])
    splitSeriesOverMultipleAxis(bow, axs, [1, 4, 7], Utils.TUM_GREEN, labelStr='Algorithm', yLim=[-0.01, 1.1])
    # splitSeriesOverMultipleAxis(userStudy, axs, [2, 7, 12], Utils.TUM_BLUE, labelStr='User Study', yLim=[-0.01, 1.1])
    # splitSeriesOverMultipleAxis(dtw, axs, [2, 5, 8], Utils.TUM_BLUE_2, labelStr='DTW', yLim=[-0.01, 1.1])
    splitSeriesOverMultipleAxis(pearson, axs, [2, 5, 8], Utils.TUM_BLUE_3, labelStr='Pearson', yLim=[-0.01, 1.1])
    # axs.flat[11].legend(['Algorithm', 'User Study', 'DTW', 'Pearson'], loc='lower center')
    allHandles = []
    allLabels = []
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        allHandles = np.concatenate([allHandles, handles])
        allLabels = np.concatenate([allLabels, labels])
    axs.flat[9].axis('off')
    axs.flat[9].legend(allHandles, allLabels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=6,fancybox=True, framealpha=1, fontsize=12)
    # leg = fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=6,fancybox=True, framealpha=1, fontsize=12)
    # axs.flat[15].legend(leg)
    # leg.set_in_Layout(True)
    # fig.tight_layout(h_pad=0.05)
    DataUtil.saveFigure(fig, 'User_Study_Eval', f'Dyad {i + 1}', 'Pearson')
