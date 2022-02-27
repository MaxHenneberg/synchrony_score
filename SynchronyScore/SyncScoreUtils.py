import numpy as np
from matplotlib import pyplot as plt
from pyts.bag_of_words import BagOfWords

from dataplayground import DataUtil


def calcOriginWordEnergy(pos, wordSize, user1):
    sum = 0
    for i in range(wordSize):
        if pos + i < len(user1):
            sum += abs(user1[pos + i])

    # Only interested in Values for User1 (We try to find the Word of User1 in a certain offset in User 2
    # So if User1 Word does not contain a lot of 0 Values, User2 doesnt aswell
    return sum / (wordSize)


def calcSyncScore(wordBinsUser1, user1, wordBinsUser2, user2, wordSize):
    syncScore = list()
    for (binsUser1, sheetUser1), (binsUser2, sheetUser2) in zip(zip(wordBinsUser1, user1), zip(wordBinsUser2, user2)):
        originalBin = binsUser1[0]
        maxSyncScore = len(binsUser1)
        syncScoreListForSheet = []
        for j, word in enumerate(originalBin):
            syncScoreForWord = 0
            for x, compareBin in enumerate(binsUser2):
                if j < len(compareBin):
                    compareWord = compareBin[j]
                    if word == compareWord:
                        energyU1 = calcOriginWordEnergy(j * wordSize, wordSize, sheetUser1)
                        energyU2 = calcOriginWordEnergy(j * wordSize + x, wordSize, sheetUser1)
                        minEnergy = min(energyU1, energyU1)
                        closeness = 0
                        if minEnergy > 0.2:
                            closeness = 1 - (abs((energyU1-energyU2)))
                        syncScoreForWord = max(0, ((maxSyncScore - x) * closeness) / maxSyncScore)
                        break

            # Add SyncScore for Current Word to List
            for x in range(wordSize):
                syncScoreListForSheet.append(syncScoreForWord)

        syncScore.append(syncScoreListForSheet)
    return syncScore


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
    maxCharacter = incrementCharacter(baseCharacter, nBins-1)
    return abs(subtractCharacter(baseCharacter, maxCharacter))

def incrementCharacter(character, increment):
    return chr(ord(character)+increment)


def subtractCharacter(cA, cB):
    return ord(cA)-ord(cB)

def wordColorImproved(word, zeroWord, nBins, wordIdx, maxIdx):
    wordDistance = 0
    maxDistance = calcMaxCharDistance(nBins)
    for cW, cZ in zip(word, zeroWord):
        # Add normalized Character Distance
        wordDistance = wordDistance + (abs(subtractCharacter(cW, cZ))/maxDistance)

    # Normalize Worddistance
    wordDistance = wordDistance / len(word)
    idxProgress = wordIdx/maxIdx
    return [wordDistance, 0, 0]


def plotSyncScore(syncScore, wordBinsUser1, wordBinsUser2, user1, user2, zeroWordList, axScales, batchSize, bow: BagOfWords,
                  targetFolder, name,
                  runId):

    for s, (sheet, (sheetWordsUser1, sheetWordsUser2)) in enumerate(
            zip(syncScore, zip(wordBinsUser1, wordBinsUser2))):
        plt.clf()
        amtOfPlots = len(sheet) // batchSize
        wordsPerBatch = batchSize // bow.word_size
        zeroWordForSheet = zeroWordList[s]
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
            for j in range(wordsPerBatch):
                startWord = (j * bow.word_size)
                endWord = min(((j + 1) * bow.word_size) + 1, batchSize - 1)
                scoreForWord = sheet[start:end][startWord]
                if scoreForWord > 0:
                    ax0.text(start + startWord, scoreForWord + 0.1, round(scoreForWord, 2), fontsize=8)
                ax0.plot(np.arange(start + startWord, start + endWord), sheet[start:end][startWord:endWord],
                         c=calcColor(scoreForWord))
            ax1 = fig.add_subplot(subplot[1, 0], sharex=ax0)
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax1.set_ylim(axScales[1])
            ax1.plot(np.arange(start, end - 1), user1[s][start:end - 1])
            startOfWords = (i * wordsPerBatch)
            endOfWords = (i * wordsPerBatch) + wordsPerBatch
            ax2 = fig.add_subplot(subplot[2, 0], sharex=ax0)
            ax2.set_ylim(axScales[2])
            ax2.plot(np.arange(start, end - 1), user2[s][start:end - 1])
            for j in range(wordsPerBatch):
                word1 = sheetWordsUser1[0][startOfWords:endOfWords][j]
                # word1Color = wordColorImproved(word1, zeroWordForSheet, bow.n_bins, j, wordsPerBatch)
                word2 = sheetWordsUser2[0][startOfWords:endOfWords][j]
                # word2Color = wordColorImproved(word2, zeroWordForSheet, bow.n_bins, j, wordsPerBatch)
                for cIdx, (c1, c2) in enumerate(zip(word1, word2)):
                    posInPlot = min(start + (j * bow.word_size) + cIdx, len(user1[s]) - 1)
                    ax1.text(posInPlot, user1[s][posInPlot] + 0.2, c1,
                             color=word1Color)
                    ax2.text(posInPlot, user2[s][posInPlot] + 0.2, c2,
                             color=word2Color)

        plt.tight_layout()
        sheetNumber = s + 1
        # There is no study 4
        if sheetNumber > 3:
            sheetNumber = sheetNumber + 1
        DataUtil.savePlot(plt, targetFolder,
                          f'{targetFolder}-S{sheetNumber}-{name}',
                          runId)

def createWordToDataMapForUser(wordUser, userData, wordMapList, wordSize):
    for i, sheet in enumerate(wordUser):
        originalBucket = sheet[0]
        map = wordMapList[i]
        for j, word in enumerate(originalBucket):
            existingData = map.get(word)
            startIdx = j * wordSize
            endIdx = startIdx + wordSize
            wordData = userData[i][startIdx:endIdx]
            if len(wordData) == wordSize:
                if existingData is not None:
                    existingData.append(list(wordData))
                    map[word] = existingData
                else:
                    map[word] = [list(wordData)]


def createWordToDataMap(wordUser1, wordsUser2, user1, user2, wordSize):
    wordMapList = [dict() for sheet in user1]
    createWordToDataMapForUser(wordUser1, user1, wordMapList, wordSize)
    createWordToDataMapForUser(wordsUser2, user2, wordMapList, wordSize)
    return wordMapList

def calcVarianz(value, average):
    return abs(value - average)

def processWordToDataMap(wordToDataMap, wordLength):
    for s, sheet in enumerate(wordToDataMap):
        print(f'Prozess Sheet {s} with {len(sheet.keys())} Words')
        numberOfWords = len(sheet.keys())
        fig, axs = plt.subplots((numberOfWords//4) + 1, 4, constrained_layout=True)
        fig.suptitle(f'Sheet: {s}', fontsize=16)
        fig.set_size_inches(20, numberOfWords // 3)
        for j, (word, dataList) in enumerate(sheet.items()):
            wordSum = 0
            summedDataList = [0 for i in range(wordLength)]
            for data in dataList:
                dataSum = 0
                for i, entry in enumerate(data):
                    summedDataList[i] = summedDataList[i]+entry
                    dataSum = dataSum + entry
                dataSum = dataSum / len(data)
                wordSum = wordSum + dataSum

            wordAverage = wordSum / len(dataList)
            varianze = 0
            averageDataList = list(np.array(summedDataList) / len(dataList))
            varianzeDataList = [0 for i in range(wordLength)]

            for data in dataList:
                dataSum = 0
                for i, entry in enumerate(data):
                    dataSum = dataSum + entry
                    varianzeDataList[i] = varianzeDataList[i]+calcVarianz(entry, averageDataList[i])

                dataSum = dataSum / len(data)
                varianze = varianze + abs(dataSum-wordAverage)

            varianze = round(varianze / len(dataList), 2)
            varianzeDataList = list(np.array(varianzeDataList) / len(dataList))

            axs.flat[j].set_title(f'{word} (Len:{len(dataList)}, Var: {varianze})')
            axs.flat[j].set_ylim([-0.5,1.5])
            axs.flat[j].plot(np.arange(wordLength), averageDataList)
            axs.flat[j].errorbar(np.arange(wordLength), averageDataList, yerr=varianzeDataList)
            for i, (avg, var) in enumerate(zip(averageDataList, varianzeDataList)):
                axs.flat[j].text(i, avg + var + 0.1, f'{round(avg, 2)}')
            for i, (avg, var) in enumerate(zip(averageDataList, varianzeDataList)):
                axs.flat[j].text(i, avg - var - 0.1, f'{round(var, 2)}')
        plt.show()

