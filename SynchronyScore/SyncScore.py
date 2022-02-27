import numbers
import operator
import string
from datetime import datetime
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from pyts.bag_of_words import BagOfWords

from SynchronyScore.BagOfWords import bowForUser, bowForTwoUsers
from SynchronyScore.SyncScoreUtils import createWordMap, mergeWordMapsForStoring, calcSyncScore, calcColor, wordColor, \
    plotSyncScore, createWordToDataMap, processWordToDataMap
from dataplayground import DataUtil
from dataplayground.DataUtil import normalizeData, createDirIfNotExistent
from utils.prepare_data import collectUserData


def calculateSyncScoreForTimeSeries(bow: BagOfWords, srcExcel: string, user1GetColumn: Callable,
                                    user2GetColumn: Callable, user1PreProcessing: Callable,
                                    user2PreProcessing: Callable, axScales: [[numbers]], targetFolder: string):
    runId = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    configSuffix = f'({bow.window_size}, {bow.word_size}, {bow.n_bins}, {bow.window_step})'
    batchSize = 256
    createDirIfNotExistent(targetFolder)

    user1, user2 = collectUserData(srcExcel, user1GetColumn,
                                   user2GetColumn)

    # Normalize Time Series
    user1 = [normalizeData(sheet) for sheet in user1]
    user2 = [normalizeData(sheet) for sheet in user2]

    # Apply Custom Preprocessing to normalized UserData
    user1 = user1PreProcessing(user1)
    user2 = user2PreProcessing(user2)

    # Perform Bag Of Words for each User
    # summedWordBinsUser1 = bowForUser(bow, user1)
    # summedWordBinsUser2 = bowForUser(bow, user2)

    summedWordBinsUser1, summedWordBinsUser2 = bowForTwoUsers(bow, user1, user2)



    # Store Words per Study in Excel
    mergedSheets = list()
    zeroWordList = list()
    for i, (sheetU1, sheetU2) in enumerate(zip(summedWordBinsUser1, summedWordBinsUser2)):
        wordMapUser1 = createWordMap(sheetU1[0])
        wordMapUser2 = createWordMap(sheetU2[0])
        mergedMap = mergeWordMapsForStoring(dict(wordMapUser1), dict(wordMapUser2))
        mergedSheets.append(mergedMap)

        # Find Zero word for coloring Words (Zero Word is (always) the most common word because of the nature of the Data
        sortedMergedMapU1 = sorted(list(wordMapUser1.items()), key=lambda kv: kv[1], reverse=True)
        sortedMergedMapU2 = sorted(list(wordMapUser2.items()), key=lambda kv: kv[1], reverse=True)
        zeroWordU1 = sortedMergedMapU1[0][0]
        zeroWordU2 = sortedMergedMapU2[0][0]
        zeroWordList.append(zeroWordU1)
        print(f'Zero Word User1/User2: {zeroWordU1}/{zeroWordU2}')

    DataUtil.saveData(["word", "user1Count", "user2Count"], mergedSheets, targetFolder, f'Words-{targetFolder}-{configSuffix}',
                      runId)


    # wordToDataMap = createWordToDataMap(summedWordBinsUser1, summedWordBinsUser2, user1, user2, bow.word_size)
    # processWordToDataMap(wordToDataMap, bow.word_size)




    # Calculate SyncScore for both directions U1->U2 and U2->U1
    syncScoreU1U2 = calcSyncScore(summedWordBinsUser1, user1, summedWordBinsUser2, user2, bow.word_size)
    syncScoreU1U2WithTimeStamp = [[np.arange(len(score)), score] for score in syncScoreU1U2]
    syncScoreU2U1 = calcSyncScore(summedWordBinsUser2, user2, summedWordBinsUser1, user1, bow.word_size)
    syncScoreU2U1WithTimeStamp = [[np.arange(len(score)), score] for score in syncScoreU2U1]

    # Store Sync Score per Direction
    # U1->U2
    DataUtil.saveData(["timestamp", "syncScore"], syncScoreU1U2WithTimeStamp, targetFolder,
                      f'SyncScoreU1U2-{targetFolder}-{configSuffix}',
                      runId)
    # U2->U1
    DataUtil.saveData(["timestamp", "syncScore"], syncScoreU2U1WithTimeStamp, targetFolder,
                      f'SyncScoreU2U1-{targetFolder}-{configSuffix}',
                      runId)

    # Average both SyncScore Directions
    syncScore = list()
    for s1, s2 in zip(syncScoreU1U2, syncScoreU2U1):
        syncScore.append((np.array(s1) + np.array(s2)) / 2)
    syncScoreWithTimeStamp = [[np.arange(len(score)), score] for score in syncScore]

    # Store averaged SyncScore
    DataUtil.saveData(["timestamp", "syncScore"], syncScoreWithTimeStamp, targetFolder,
                      f'SyncScore-{targetFolder}-{configSuffix}',
                      runId)

    # Plot SyncScore on top of User1/2-Plot including origin Words for each User
    plotSyncScore(syncScore, summedWordBinsUser1, summedWordBinsUser2, user1, user2, zeroWordList, axScales, batchSize,
                  bow,
                  targetFolder, configSuffix,
                  runId)
