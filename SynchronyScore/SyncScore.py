import _thread
import numbers
import string
from datetime import datetime
from typing import Callable

import numpy as np
from pyts.bag_of_words import BagOfWords

from SynchronyScore.BagOfWords import bowForTwoUsers
from SynchronyScore.SyncScoreUtils import createWordMap, mergeWordMapsForStoring, calcSyncScore, \
    plotSyncScore, createWordToDataMap, processWordToDataMap, calcGaussianScynScore, createVarianceList, \
    countSyncSections
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

    summedWordBinsUser1, summedWordBinsUser2 = bowForTwoUsers(bow, user1, user2)

    # Store Words per Study in Excel
    mergedSheets = list()
    for i, (sheetU1, sheetU2) in enumerate(zip(summedWordBinsUser1, summedWordBinsUser2)):
        wordMapUser1 = createWordMap(sheetU1[0])
        wordMapUser2 = createWordMap(sheetU2[0])
        mergedMap = mergeWordMapsForStoring(dict(wordMapUser1), dict(wordMapUser2))
        mergedSheets.append(mergedMap)

    DataUtil.saveData(["word", "user1Count", "user2Count"], mergedSheets, targetFolder,
                      f'Words-{targetFolder}-{configSuffix}',
                      runId)

    wordToDataMap = createWordToDataMap(summedWordBinsUser1, summedWordBinsUser2, user1, user2, bow.window_size)
    processWordToDataMap(wordToDataMap, bow.window_size, targetFolder, configSuffix, runId)

    # Calculate SyncScore for both directions U1->U2 and U2->U1
    syncScoreU1U2 = calcGaussianScynScore(summedWordBinsUser1, user1, summedWordBinsUser2, user2, bow.window_size,
                                          bow.n_bins)
    syncScoreU1U2WithTimeStamp = [[np.arange(len(score)), score] for score in syncScoreU1U2]
    syncScoreU2U1 = calcGaussianScynScore(summedWordBinsUser2, user2, summedWordBinsUser1, user1, bow.window_size,
                                          bow.n_bins)
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
    plotSyncScore(syncScore, summedWordBinsUser1, summedWordBinsUser2, user1, user2, axScales, batchSize,
                  bow,
                  targetFolder, configSuffix,
                  runId)


def calcSyncScoreAndVarianceForComparison(bow: BagOfWords, srcExcel: string, user1GetColumn: Callable,
                                          user2GetColumn: Callable, user1PreProcessing: Callable,
                                          user2PreProcessing: Callable, axScales: [[numbers]], targetFolder: string,
                                          printDiagrams: bool, runId):
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

    summedWordBinsUser1, summedWordBinsUser2 = bowForTwoUsers(bow, user1, user2)

    # Store Words per Study in Excel
    mergedSheets = list()
    for i, (sheetU1, sheetU2) in enumerate(zip(summedWordBinsUser1, summedWordBinsUser2)):
        wordMapUser1 = createWordMap(sheetU1[0])
        wordMapUser2 = createWordMap(sheetU2[0])
        mergedMap = mergeWordMapsForStoring(dict(wordMapUser1), dict(wordMapUser2))
        mergedSheets.append(mergedMap)
    if printDiagrams:
        DataUtil.saveData(["word", "user1Count", "user2Count"], mergedSheets, targetFolder,
                          f'Words-{targetFolder}-{configSuffix}',
                          runId)

    wordToDataMap = createWordToDataMap(summedWordBinsUser1, summedWordBinsUser2, user1, user2, bow.window_size)
    if printDiagrams:
        processWordToDataMap(wordToDataMap, bow.window_size, targetFolder, configSuffix, runId)

    variancePerSheetList = createVarianceList(wordToDataMap, bow.window_size)

    # Calculate SyncScore for both directions U1->U2 and U2->U1
    syncScoreU1U2 = calcGaussianScynScore(summedWordBinsUser1, user1, summedWordBinsUser2, user2, bow.window_size,
                                          bow.n_bins)
    syncScoreU1U2WithTimeStamp = [[np.arange(len(score)), score] for score in syncScoreU1U2]
    syncScoreU2U1 = calcGaussianScynScore(summedWordBinsUser2, user2, summedWordBinsUser1, user1, bow.window_size,
                                          bow.n_bins)
    syncScoreU2U1WithTimeStamp = [[np.arange(len(score)), score] for score in syncScoreU2U1]

    # Store Sync Score per Direction
    # U1->U2
    if printDiagrams:
        DataUtil.saveData(["timestamp", "syncScore"], syncScoreU1U2WithTimeStamp, targetFolder,
                          f'SyncScoreU1U2-{targetFolder}-{configSuffix}',
                          runId)
        # U2->U1
        DataUtil.saveData(["timestamp", "syncScore"], syncScoreU2U1WithTimeStamp, targetFolder,
                          f'SyncScoreU2U1-{targetFolder}-{configSuffix}',
                          runId)

    # Average both SyncScore Directions
    syncScorePerSheet = list()
    for s1, s2 in zip(syncScoreU1U2, syncScoreU2U1):
        syncScorePerSheet.append((np.array(s1) + np.array(s2)) / 2)

    syncSectionsPerSheet = list()
    for sheetSyncScore in syncScorePerSheet:
        syncSectionsPerSheet.append(countSyncSections(sheetSyncScore))

    syncScoreWithTimeStamp = [[np.arange(len(score)), score] for score in syncScorePerSheet]

    # Store averaged SyncScore
    if printDiagrams:
        DataUtil.saveData(["timestamp", "syncScore"], syncScoreWithTimeStamp, targetFolder,
                          f'SyncScore-{targetFolder}-{configSuffix}',
                          runId)

    # Plot SyncScore on top of User1/2-Plot including origin Words for each User
    if printDiagrams:
        plotSyncScore(syncScorePerSheet, summedWordBinsUser1, summedWordBinsUser2, user1, user2, axScales, batchSize,
                      bow,
                      targetFolder, configSuffix,
                      runId)

    return syncScorePerSheet, variancePerSheetList, syncSectionsPerSheet
