from datetime import datetime

import numpy as np
from pyts.bag_of_words import BagOfWords

from SynchronyScore.SyncScore import calcSyncScoreAndVarianceForComparison
from SynchronyScore.SyncScoreUtils import plotComparisonResults
from dataplayground.DataUtil import normalizeData

windowSize = 1
wordSize = 2
nBins = 2
windowStep = 1

iterations = 5
iterationIncrease = 2
baseTitle = 'WindowSize'
sheetAmount = 9

configSuffix = f'({windowSize}, {wordSize}, {nBins}, {windowStep})'
runTitle = f'{baseTitle} {configSuffix}'
targetFolder = f'COMPARE_CONFIG_WITH_SYNC_SECTION_{baseTitle}'
runId = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')

configValuesPerRun = list()
syncScorePerRunPerSheet = list()
variancePerRunPerSheet = list()
syncSectionPerRunPerSheet = list()
for i in range(iterations):
    # Increase variable for this run
    windowSize = windowSize * iterationIncrease

    configValuesPerRun.append(windowSize)
    print(f'Iteration {i + 1} with Value {windowSize}')

    bow = BagOfWords(window_size=windowSize, word_size=wordSize,
                     window_step=windowStep, numerosity_reduction=False, n_bins=nBins, strategy='uniform')
    plotResults = False
    # if i == 0 or i == (iterations - 1):
    #     plotResults = True

    syncScoreListPerSheet, variancePerSheet, syncSectionsPerSheet = calcSyncScoreAndVarianceForComparison(bow, "Smile",
                                                                                                          lambda sheet:
                                                                                                          sheet[
                                                                                                              "User1Smile"],
                                                                                                          lambda sheet:
                                                                                                          sheet[
                                                                                                              "User2Smile"],
                                                                                                          lambda
                                                                                                              data: data,
                                                                                                          lambda
                                                                                                              data: data,
                                                                                                          [[-0.1, 1.5],
                                                                                                           [-0.1, 1.5],
                                                                                                           [-0.1, 1.5]],
                                                                                                          targetFolder,
                                                                                                          plotResults,
                                                                                                          runId)

    syncScorePerSheet = list()
    for syncScoreList in syncScoreListPerSheet:
        syncScorePerSheet.append(sum(syncScoreList) / len(syncScoreList))

    syncScorePerRunPerSheet.append(syncScorePerSheet)
    variancePerRunPerSheet.append(variancePerSheet)
    syncSectionPerRunPerSheet.append(syncSectionsPerSheet)

syncScorePerSheetPerRun = [[] for i in range(9)]
variancePerSheetPerRun = [[] for i in range(9)]
syncSectionsPerSheetPerRun = [[] for i in range(9)]
for syncScoreRun, varianceRun, syncSectionRun in zip(syncScorePerRunPerSheet, variancePerRunPerSheet,
                                                     syncSectionPerRunPerSheet):
    for sheetIdx, (syncScoreSheet, varianceSheet, syncSectionSheet) in enumerate(
            zip(syncScoreRun, varianceRun, syncSectionRun)):
        syncScorePerSheetPerRun[sheetIdx].append(syncScoreSheet)
        variancePerSheetPerRun[sheetIdx].append(varianceSheet)
        syncSectionsPerSheetPerRun[sheetIdx].append(syncSectionSheet)

syncScorePerSheetPerRunNormalized = list()
variancePerSheetPerRunNormalized = list()
syncSectionsPerSheetPerRunNormalized = list()
for syncScore, variance, syncSection in zip(syncScorePerSheetPerRun, variancePerSheetPerRun,
                                            syncSectionsPerSheetPerRun):
    syncScorePerSheetPerRunNormalized.append(np.array(normalizeData(syncScore)))
    variancePerSheetPerRunNormalized.append(np.array(normalizeData(variance)))
    syncSectionsPerSheetPerRunNormalized.append(np.array(normalizeData(syncSection)))

mergedSyncScorePerRun = list()
mergedVariancePerRun = list()
mergedSyncSectionsPerRun = list()
for syncScoreList, varianceList, syncSectionList in zip(syncScorePerRunPerSheet, variancePerRunPerSheet,
                                                        syncSectionPerRunPerSheet):
    lenSyncScore = len(syncScoreList)
    lenVarScore = len(varianceList)
    lenSyncSection = len(syncSectionList)
    mergedSyncScorePerRun.append(sum(syncScoreList) / lenSyncScore)
    mergedVariancePerRun.append(sum(varianceList) / lenVarScore)
    mergedSyncSectionsPerRun.append(sum(syncSectionList) / lenSyncSection)

mergedSyncScoreNormalized = [0 for i in range(len(configValuesPerRun))]
mergedVarianceNormalized = [0 for i in range(len(configValuesPerRun))]
mergedSyncSectionsNormalized = [0 for i in range(len(configValuesPerRun))]
for syncScorePerRun, variancePerRun, syncSectionPerRun in zip(syncScorePerSheetPerRunNormalized,
                                                              variancePerSheetPerRunNormalized,
                                                              syncSectionsPerSheetPerRunNormalized):
    for i, (syncScoreOfRun, varianceOfRun, syncSectionOfRun) in enumerate(
            zip(syncScorePerRun, variancePerRun, syncSectionPerRun)):
        mergedSyncScoreNormalized[i] = mergedSyncScoreNormalized[i] + syncScoreOfRun
        mergedVarianceNormalized[i] = mergedVarianceNormalized[i] + varianceOfRun
        mergedSyncSectionsNormalized[i] = mergedSyncSectionsNormalized[i] + syncSectionOfRun

mergedSyncScoreNormalized = np.array(mergedSyncScoreNormalized) / sheetAmount
mergedVarianceNormalized = np.array(mergedVarianceNormalized) / sheetAmount
mergedSyncSectionsNormalized = np.array(mergedSyncSectionsNormalized) / sheetAmount

plotComparisonResults(syncScorePerSheetPerRun, variancePerSheetPerRun, syncSectionsPerSheetPerRun,
                      configValuesPerRun,
                      mergedSyncScorePerRun, mergedVariancePerRun, mergedSyncSectionsPerRun,
                      syncScorePerSheetPerRunNormalized, variancePerSheetPerRunNormalized, syncSectionsPerSheetPerRunNormalized,
                      mergedSyncScoreNormalized, mergedVarianceNormalized, mergedSyncSectionsNormalized,
                      targetFolder, runTitle, runId)
