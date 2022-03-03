import _thread
import cmath
import math

import numpy as np
from matplotlib import pyplot as plt
from pyts.bag_of_words import BagOfWords

from dataplayground import DataUtil
from dataplayground.DataUtil import normalizeData, calcSheetNumber


def calcOriginWordEnergy(pos, wordSize, user1):
    sum = 0
    for i in range(wordSize):
        if pos + i < len(user1):
            sum += abs(user1[pos + i])

    # Only interested in Values for User1 (We try to find the Word of User1 in a certain offset in User 2
    # So if User1 Word does not contain a lot of 0 Values, User2 doesnt aswell
    return sum / (wordSize)


def calcSyncScore(wordBinsUser1, user1, wordBinsUser2, user2, windowSize):
    syncScore = list()
    for (binsUser1, sheetUser1), (binsUser2, sheetUser2) in zip(zip(wordBinsUser1, user1), zip(wordBinsUser2, user2)):
        originalBin = binsUser1[0]
        maxSyncScore = len(binsUser1)
        syncScoreListForSheet = list()
        for j, word in enumerate(originalBin):
            syncScoreForWord = 0
            for x, compareBin in enumerate(binsUser2):
                if j < len(compareBin):
                    compareWord = compareBin[j]
                    if word == compareWord:
                        energyU1 = calcOriginWordEnergy(j * windowSize, windowSize, sheetUser1)
                        energyU2 = calcOriginWordEnergy(j * windowSize + x, windowSize, sheetUser2)
                        minEnergy = min(energyU1, energyU2)
                        closeness = 0
                        if minEnergy > 0.2:
                            closeness = 1 - (abs((energyU1 - energyU2)))
                        syncScoreForWord = max(0, ((maxSyncScore - x) * closeness) / maxSyncScore)
                        break

            # Add SyncScore for Current Word to List
            for x in range(windowSize):
                syncScoreListForSheet.append(syncScoreForWord)

        syncScore.append(syncScoreListForSheet)
    return syncScore


def calcGaussianScynScore(wordBinsUser1, user1, wordBinsUser2, user2, windowSize, alphabetSize):
    syncScore = list()
    for (binsUser1, sheetUser1), (binsUser2, sheetUser2) in zip(zip(wordBinsUser1, user1), zip(wordBinsUser2, user2)):
        originalBin = binsUser1[0]
        maxSyncScore = (len(binsUser1) + 1) * (len(binsUser1) / 2)
        syncScoreListForSheet = list()
        for j, word in enumerate(originalBin):
            syncScoreForWord = 0
            energyU1 = calcOriginWordEnergy(j * windowSize, windowSize, sheetUser1)
            energyU2 = calcOriginWordEnergy(j * windowSize, windowSize, sheetUser2)
            minEnergy = min(energyU1, energyU2)
            closeness = 1 - (abs((energyU1 - energyU2)))
            if minEnergy > 0.1:
                for x, compareBin in enumerate(binsUser2):
                    if j < len(compareBin):
                        compareWord = compareBin[j]
                        gaussianWordFactor = 1 - calcGaussianWordDistance(word, compareWord, alphabetSize)
                        syncScoreForWord = syncScoreForWord + max(0,
                                                                  (len(binsUser2) - x) * gaussianWordFactor * closeness)
            # Normalize Sync Score
            syncScoreForWord = syncScoreForWord / maxSyncScore
            # Add SyncScore for Current Word to List
            for x in range(windowSize):
                syncScoreListForSheet.append(syncScoreForWord)

        syncScore.append(syncScoreListForSheet)
    return syncScore


def calcGaussianWordDistance(word, compareWord, alphabetSize):
    distanceSum = 0
    for c1, c2 in zip(word, compareWord):
        # Normalize Distance per Character
        charDistance = (abs(ord(c1) - ord(c2))) / alphabetSize
        distanceSum = distanceSum + charDistance
    # Normalize distance per Word
    return distanceSum / len(word)


def createWordMap(originalWords):
    wordMap = dict()
    for word in originalWords:
        wordCount = wordMap.get(word)
        if (wordCount is not None):
            wordCount = wordCount + 1
            wordMap[word] = wordCount
        else:
            wordMap[word] = 1

    return wordMap


def mergeWordMapsForStoring(user1Map: dict, user2Map: dict):
    toBeStored = [list(), list(), list()]
    for word in user1Map.keys():
        user1 = user1Map.get(word)
        user2 = user2Map.get(word)
        toBeStored[0].append(word)
        toBeStored[1].append(user1)
        if user2 is not None:
            toBeStored[2].append(user2)
            user2Map.pop(word)
        else:
            toBeStored[2].append(0)

    for word in user2Map.keys():
        user2 = user2Map.get(word)
        toBeStored[0].append(word)
        toBeStored[1].append(0)
        toBeStored[2].append(user2)

    return toBeStored


def minMaxColor(value):
    return max(0, min(1, value))


def calcColor(scoreForWord):
    return [minMaxColor(1 - scoreForWord), minMaxColor(scoreForWord), 0]


def wordColor(word):
    r = 0
    b = 0
    for i, c in enumerate(word):
        if c == 'a':
            r += (i * i)
        else:
            b += (i * i)

    return [r / 255, 0, b / 255]


def calcMaxCharDistance(nBins):
    baseCharacter = 'a'
    maxCharacter = incrementCharacter(baseCharacter, nBins - 1)
    return abs(subtractCharacter(baseCharacter, maxCharacter))


def incrementCharacter(character, increment):
    return chr(ord(character) + increment)


def subtractCharacter(cA, cB):
    return ord(cA) - ord(cB)


def wordColorImproved(word, zeroWord, nBins, wordIdx, maxIdx):
    wordDistance = 0
    maxDistance = calcMaxCharDistance(nBins)
    for cW, cZ in zip(word, zeroWord):
        # Add normalized Character Distance
        wordDistance = wordDistance + (abs(subtractCharacter(cW, cZ)) / maxDistance)

    # Normalize Worddistance
    wordDistance = wordDistance / len(word)
    idxProgress = wordIdx / maxIdx
    return [wordDistance, 0, 0]


def plotSyncScore(syncScore, wordBinsUser1, wordBinsUser2, user1, user2, axScales, batchSize, bow: BagOfWords,
                  targetFolder, name,
                  runId):
    for s, (sheet, (sheetWordsUser1, sheetWordsUser2)) in enumerate(
            zip(syncScore, zip(wordBinsUser1, wordBinsUser2))):
        plotSheet(
            s, sheet, sheetWordsUser1, sheetWordsUser2, user1, user2, axScales, batchSize, bow, targetFolder, name,
            runId)


def plotSheet(s, sheet, sheetWordsUser1, sheetWordsUser2, user1, user2, axScales, batchSize, bow: BagOfWords,
              targetFolder, name, runId):
    amtOfPlots = len(sheet) // batchSize
    wordsPerBatch = batchSize // bow.window_size
    sheetNumber = calcSheetNumber(s)
    print(f'Start SyncScorePlot S{sheetNumber}')
    fig = plt.figure()
    fig.set_size_inches(20, 70)
    gs0 = fig.add_gridspec(amtOfPlots, 1)
    for i in range(amtOfPlots):
        subplot = gs0[i].subgridspec(3, 1, hspace=0)
        start = (i * batchSize)
        end = min(((i + 1) * batchSize) + 1, len(sheet) - 1)
        ax0 = fig.add_subplot(subplot[0, 0])
        plt.setp(ax0.get_xticklabels(), visible=False)
        ax0.set_ylim(axScales[0])
        # User 1
        ax1 = fig.add_subplot(subplot[1, 0], sharex=ax0)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_ylim(axScales[1])
        # User 2
        ax2 = fig.add_subplot(subplot[2, 0], sharex=ax0)
        ax2.set_ylim(axScales[2])
        for j in range(wordsPerBatch):
            # Start of Words in Data
            startWord = (j * bow.window_size)
            # End of Words in Data
            endWord = min(((j + 1) * bow.window_size) + 1, batchSize - 1)
            # First syncValue for Sheet
            scoreForWord = sheet[start:end][startWord]
            # Only plot SyncScore value if > 0
            if scoreForWord > 0:
                ax0.text(start + startWord, scoreForWord + 0.1, round(scoreForWord, 2), fontsize=8)
            # Plot Sync Score
            ax0.plot(np.arange(start + startWord, start + endWord), sheet[start:end][startWord:endWord],
                     c=calcColor(scoreForWord))
            # Plot first Batch of User1
            ax1.plot(np.arange(start + startWord, start + endWord), user1[s][start:end][startWord:endWord],
                     color="C{}".format(j % 2))
            # Plot first Batch of User2
            ax2.plot(np.arange(start + startWord, start + endWord), user2[s][start:end][startWord:endWord],
                     color="C{}".format(j % 2))

        startOfWords = (i * wordsPerBatch)
        endOfWords = (i * wordsPerBatch) + wordsPerBatch
        for j in range(wordsPerBatch):
            word1 = sheetWordsUser1[0][startOfWords:endOfWords][j]
            # word1Color = wordColorImproved(word1, zeroWordForSheet, bow.n_bins, j, wordsPerBatch)
            word2 = sheetWordsUser2[0][startOfWords:endOfWords][j]
            # word2Color = wordColorImproved(word2, zeroWordForSheet, bow.n_bins, j, wordsPerBatch)
            for cIdx, (c1, c2) in enumerate(zip(word1, word2)):
                posInPlot = min(start + (j * bow.window_size) + cIdx, len(user1[s]) - 1)
                ax1.text(posInPlot, user1[s][posInPlot] + 0.2, c1,
                         color="C{}".format(j % 2))
                ax2.text(posInPlot, user2[s][posInPlot] + 0.2, c2,
                         color="C{}".format(j % 2))

    fig.tight_layout()
    DataUtil.saveFigure(fig, targetFolder,
                        f'{targetFolder}-S{sheetNumber}-{name}',
                        runId)
    fig.clear()
    print(f'Finished SyncScorePlot S{sheetNumber}')


def createWordToDataMapForUser(wordUser, userData, wordMapList, windowSize):
    for i, sheet in enumerate(wordUser):
        originalBucket = sheet[0]
        map = wordMapList[i]
        for j, word in enumerate(originalBucket):
            existingData = map.get(word)
            startIdx = j * windowSize
            endIdx = startIdx + windowSize
            wordData = normalizeData(userData[i][startIdx:endIdx])
            if len(wordData) == windowSize:
                if existingData is not None:
                    existingData.append(list(wordData))
                    map[word] = existingData
                else:
                    map[word] = [list(wordData)]


def createWordToDataMap(wordUser1, wordsUser2, user1, user2, windowSize):
    wordMapList = [dict() for sheet in user1]
    createWordToDataMapForUser(wordUser1, user1, wordMapList, windowSize)
    createWordToDataMapForUser(wordsUser2, user2, wordMapList, windowSize)
    return wordMapList


def calcVarianz(value, average):
    return abs(value - average)


def createVarianceList(wordToDataMap, windowSize):
    varianceList = list()
    for s, sheet in enumerate(wordToDataMap):
        sortedKeys = sorted(sheet.keys())
        sheetVariance = 0
        for j, word in enumerate(sortedKeys):
            wordSum = 0
            summedDataList = [0 for i in range(windowSize)]
            dataList = sheet[word]
            for data in dataList:
                dataSum = 0
                for i, entry in enumerate(data):
                    summedDataList[i] = summedDataList[i] + entry
                    dataSum = dataSum + entry
                dataSum = dataSum / len(data)
                wordSum = wordSum + dataSum

            wordAverage = wordSum / len(dataList)
            variance = 0
            averageDataList = list(np.array(summedDataList) / len(dataList))
            varianzeDataList = [0 for i in range(windowSize)]

            for data in dataList:
                dataSum = 0
                for i, entry in enumerate(data):
                    dataSum = dataSum + entry
                    varianzeDataList[i] = varianzeDataList[i] + calcVarianz(entry, averageDataList[i])

                dataSum = dataSum / len(data)
                variance = variance + abs(dataSum - wordAverage)

            sheetVariance = sheetVariance + round(variance / len(dataList), 2)

        varianceList.append(sheetVariance / len(sortedKeys))
    return varianceList


def processWordToDataMap(wordToDataMap, windowSize, targetFolder, name, runId):
    for s, sheet in enumerate(wordToDataMap):
        processWordToDataMapSheet(s, sheet, windowSize, targetFolder, name, runId)


def processWordToDataMapSheet(s, sheet, windowSize, targetFolder, name, runId):
    sheetNumber = calcSheetNumber(s)
    print(f'Prozess Sheet {sheetNumber} with {len(sheet.keys())} Words')
    columns = 4
    numberOfWords = len(sheet.keys())
    if numberOfWords // 4 < numberOfWords / 4:
        numberOfRows = numberOfWords // 4 + 1
    else:
        numberOfRows = numberOfWords // 4
    fig, axs = plt.subplots(numberOfRows, columns, constrained_layout=True)
    fig.suptitle(f'Sheet: {sheetNumber}', fontsize=16)
    fig.set_size_inches(5 * columns, 5 * numberOfRows)
    sortedKeys = sorted(sheet.keys())
    for j, word in enumerate(sortedKeys):
        wordSum = 0
        summedDataList = [0 for i in range(windowSize)]
        dataList = sheet[word]
        for data in dataList:
            dataSum = 0
            for i, entry in enumerate(data):
                summedDataList[i] = summedDataList[i] + entry
                dataSum = dataSum + entry
            dataSum = dataSum / len(data)
            wordSum = wordSum + dataSum

        wordAverage = wordSum / len(dataList)
        varianze = 0
        averageDataList = list(np.array(summedDataList) / len(dataList))
        varianzeDataList = [0 for i in range(windowSize)]

        for data in dataList:
            dataSum = 0
            for i, entry in enumerate(data):
                dataSum = dataSum + entry
                varianzeDataList[i] = varianzeDataList[i] + calcVarianz(entry, averageDataList[i])

            dataSum = dataSum / len(data)
            varianze = varianze + abs(dataSum - wordAverage)

        varianze = round(varianze / len(dataList), 2)
        varianzeDataList = list(np.array(varianzeDataList) / len(dataList))

        axs.flat[j].set_title(f'{word} (Occurrences: {len(dataList)}, Word Var: {varianze})')
        axs.flat[j].set_ylim([-0.2, 1.2])
        axs.flat[j].plot(np.arange(windowSize), averageDataList)
        axs.flat[j].errorbar(np.arange(windowSize), averageDataList, yerr=varianzeDataList)
        for i, (avg, var) in enumerate(zip(averageDataList, varianzeDataList)):
            axs.flat[j].text(i, avg + var + 0.1, f'{round(avg, 2)}')
        for i, (avg, var) in enumerate(zip(averageDataList, varianzeDataList)):
            axs.flat[j].text(i, avg - var - 0.1, f'{round(var, 2)}')

    DataUtil.saveFigure(fig, targetFolder,
                        f'{targetFolder}-S{sheetNumber}-{name}-WordSummary',
                        runId)
    fig.clear()
    print(f'Finished Sheet {sheetNumber}')


def plotComparisonResults(syncScorePerSheet, variancePerSheet, syncSectionsPerSheet,
                          configValues,
                          mergedSyncScore, mergedVariance, mergedSyncSectionsPerRun,
                          syncScoreNormalized, varianceNormalized, syncSectionsNormalized,
                          mergedSyncScoreNormalized, mergedVarianceNormalized, mergedSyncSectionsNormalized,
                          targetFolder, runTitle, runId):
    fig, axs = plt.subplots(10, 2, constrained_layout=True)
    fig.suptitle('', fontsize=16)
    fig.set_size_inches(10, 30)
    axs.flat[0].set_title(f'Merged Values')
    axs.flat[0].plot(configValues, mergedSyncScore, label='SyncScore', marker="o", color="C0")
    axs.flat[0].set_xticks(configValues)
    axs.flat[0].set_xticklabels(configValues)
    axs.flat[0].plot(configValues, mergedVariance, label='Variance', marker="o", color="C1")
    axs0Twin = axs.flat[0].twinx()
    axs0Twin.plot(configValues, mergedSyncSectionsPerRun, label='SyncSection', marker="o", color="C2")
    # axs0Twin.legend(loc='upper left', bbox_to_anchor=(-1, 0.5))
    # axs.flat[0].legend(loc='lower left', bbox_to_anchor=(-1, 0.5))
    axs.flat[1].set_title(f'Merged Values Normalized')
    axs.flat[1].plot(configValues, mergedSyncScoreNormalized, label='SyncScore', marker="o", color="C0")
    axs.flat[1].set_xticks(configValues)
    axs.flat[1].set_xticklabels(configValues)
    axs.flat[1].plot(configValues, mergedVarianceNormalized, label='Variance', marker="o", color="C1")
    axs1Twin = axs.flat[1].twinx()
    axs1Twin.plot(configValues, mergedSyncSectionsNormalized, label='SyncSection', marker="o", color="C2")

    handles, labels = axs.flat[1].get_legend_handles_labels()
    twinHandles, twinLabels = axs1Twin.get_legend_handles_labels()
    mergedLabels = labels + twinLabels
    axs.flat[1].legend(handles + twinHandles, mergedLabels, loc='center left',
               bbox_to_anchor=(1.13, 0.5))
    # axs1Twin.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    # axs.flat[1].legend(loc='lower left', bbox_to_anchor=(1, 0.5))
    for (i, mergedSyncValue, mergedVarianceValue, mergedSyncSectionValue, mergedSyncValueNormalized,
         mergedVaraianceValueNormalized, mergedSyncSectionValueNormalized) in zip(
        configValues, mergedSyncScore, mergedVariance, mergedSyncSectionsPerRun, mergedSyncScoreNormalized,
        mergedVarianceNormalized, mergedSyncSectionsNormalized):
        axs.flat[0].text(i, mergedSyncValue * 1.03, f'{round(mergedSyncValue, 3)}')
        axs.flat[0].text(i, mergedVarianceValue * 1.03, f'{round(mergedVarianceValue, 3)}')
        axs0Twin.text(i, mergedSyncSectionValue * 1.03, f'{round(mergedSyncSectionValue, 3)}')
        axs.flat[1].text(i, mergedSyncValueNormalized * 1.03, f'{round(mergedSyncValueNormalized, 3)}')
        axs.flat[1].text(i, mergedVaraianceValueNormalized * 1.03, f'{round(mergedVaraianceValueNormalized, 3)}')
        axs1Twin.text(i, mergedSyncSectionValueNormalized * 1.03, f'{round(mergedSyncSectionValueNormalized, 3)}')

    for i, (
            syncScore, variance, syncSection, syncScoreValueNormed, varianceValueNormed,
            syncSectionValueNormed) in enumerate(
        zip(syncScorePerSheet, variancePerSheet, syncSectionsPerSheet, syncScoreNormalized, varianceNormalized,
            syncSectionsNormalized)):
        axsIdx = 2 * (i + 1)
        sheetNumber = calcSheetNumber(i)
        axs.flat[axsIdx].set_title(f'Sheet: {sheetNumber}')
        axs.flat[axsIdx].plot(configValues, syncScore, label='SyncScore', marker="o", color="C0")
        axs.flat[axsIdx].plot(configValues, variance, label='Variance', marker="o", color="C1")
        axsTwin0 = axs.flat[axsIdx].twinx()
        axsTwin0.plot(configValues, syncSection, label='SyncSection', marker="o", color="C2")
        axs.flat[axsIdx].set_xticks(configValues)
        axs.flat[axsIdx].set_xticklabels(configValues)
        # axs.flat[axsIdx].legend(loc='upper left', bbox_to_anchor=(-1, 0.5))
        # axsTwin0.legend(loc='lower left', bbox_to_anchor=(-1, 0.5))
        for (j, syncScoreValue, varianceValue, syncSectionValue) in zip(configValues, syncScore, variance, syncSection):
            axs.flat[axsIdx].text(j, syncScoreValue * 1.03, f'{round(syncScoreValue, 3)}')
            axs.flat[axsIdx].text(j, varianceValue * 1.03, f'{round(varianceValue, 3)}')
            axsTwin0.text(j, syncSectionValue * 1.03, f'{round(syncSectionValue, 3)}')

        axs.flat[axsIdx + 1].set_title(f'Sheet: {sheetNumber} Normalized')
        axs.flat[axsIdx + 1].plot(configValues, syncScoreValueNormed, label='SyncScore', marker="o", color="C0")
        axs.flat[axsIdx + 1].plot(configValues, varianceValueNormed, label='Variance', marker="o", color="C1")
        axsTwin1 = axs.flat[axsIdx + 1].twinx()
        axsTwin1.plot(configValues, syncSectionValueNormed, label='SyncSection', marker="o", color="C2")
        axs.flat[axsIdx + 1].set_xticks(configValues)
        axs.flat[axsIdx + 1].set_xticklabels(configValues)
        # axs.flat[axsIdx + 1].legend(loc='upper left', bbox_to_anchor=(1, 0.5))
        # axsTwin1.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
        for (j, mergedSyncValue, mergedVarianceValue, ssVN) in zip(configValues, syncScoreValueNormed,
                                                                   varianceValueNormed, syncSectionValueNormed):
            axs.flat[axsIdx + 1].text(j, mergedSyncValue * 1.03, f'{round(mergedSyncValue, 3)}')
            axs.flat[axsIdx + 1].text(j, mergedVarianceValue * 1.03, f'{round(float(mergedVarianceValue), 3)}')
            axsTwin1.text(j, ssVN * 1.03, f'{round(ssVN, 3)}')

    DataUtil.saveFigure(fig, targetFolder,
                        f'{targetFolder}-{runTitle}-WordSummary',
                        runId)


def countSyncSections(syncScore):
    count = 0
    prevValue = syncScore[0]
    for value in syncScore:
        if prevValue == 0 and value > 0:
            count = count + 1
        prevValue = value

    return count
