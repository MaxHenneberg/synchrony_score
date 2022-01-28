import math
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from pyts.bag_of_words import BagOfWords

from dataplayground import DataUtil
from dataplayground.DataUtil import normalizeData, slidingWindow


def calcColorSum(word: str):
    sum = 0
    for i, c in enumerate(word.lower()):
        if c not in ['a', 'b']:
            sum += (ord(c) - 96) * i
        else:
            sum += 1

    return sum


def calcColor(sum, maxColor):
    return [max(0.0, min(0.3, (float(sum) / float(maxColor)))) / 0.35,
            max(0.0, min(0.3, (float(sum) / float(maxColor) - 0.3))) / 0.35,
            max(0.0, min(0.3, (float(sum) / float(maxColor) - 0.6))) / 0.35]


def bowOfBatch(bow: BagOfWords, batch, maxColor, bowList, bowMap):
    bowForBatch = bow.transform([batch])
    bowList.append(bowForBatch)
    wordCount = Counter(bowForBatch[0].split(' '))
    for key in bowMap.keys():
        mergedVal = bowMap.get(key)
        if key in wordCount:
            mergedVal['count'] += wordCount.pop(key)
        bowMap[key] = mergedVal

    for key in wordCount.keys():
        nColor = calcColorSum(key)
        maxColor = max(nColor, maxColor)
        nObject = {key: {'count': wordCount.get(key), 'color': nColor, 'sample': []}}
        bowMap.update(nObject)
    return maxColor


def printBow(bowList, bowMap, batches, maxColor, wordSize, batchSize, titles, imageFolder, imageName):
    fig, axs = plt.subplots(len(bowList))
    fig.set_size_inches(30, 70)
    # Plot the corresponding letters
    for j in range(len(bowList)):
        batchUser1Bow = bowList[j]
        for i, word in enumerate(batchUser1Bow[0].split(' ')):
            wordObj = bowMap.get(word)
            color = wordObj['color']
            start = (i * wordSize)
            end = min(((i + 1) * wordSize) + 1, batchSize)
            if len(wordObj['sample']) == wordSize:
                wordObj['sample'] = (wordObj['sample'] + batches[j][start:end][0:wordSize])
            else:
                wordObj['sample'] = batches[j][start:end][0:wordSize]

            bowMap[word] = wordObj
            axs[j].set_ylim([-0.1, 1.1])
            if len(titles) == len(bowList):
                axs[j].set_title(titles[j])
            axs[j].plot(np.arange(start, end), batches[j][start:end], 'o-', color=calcColor(color, maxColor),
                        lw=1,
                        ms=1)
            for x, c in enumerate(word):
                axs[j].text(start + x, batches[j][start + x] + 0.1, c, color=calcColor(color, maxColor),
                            fontsize=14)

    plt.tight_layout()
    DataUtil.savePlot(plt, imageFolder, imageName)


def bagOfWordsForSheet(bow: BagOfWords, sheet, batchSize, userNumber, sheetNumber, bowMap, maxColor, folder):
    print(f'Processing Sheet {sheetNumber} of User {userNumber}')
    sheetBatched = slidingWindow(normalizeData(sheet), batchSize, batchSize)
    bowList = list()
    bowMap = bowMap
    maxColor = maxColor
    for batch in sheetBatched:
        maxColor = bowOfBatch(bow, batch, maxColor, bowList, bowMap)

    printBow(bowList, bowMap, sheetBatched, maxColor, bow.word_size, batchSize, [], folder,
             f'User-{userNumber}_Sheet-{sheetNumber}')
    print(f'Finished Sheet {sheetNumber} of User {userNumber}')
    return maxColor, bowList

def splitWordsIntoBins(bowList, nBins: int):
    bins = []
    for i in range(nBins):
        bins.append([])

    for i, word in enumerate(bowList):
        bin = bins.pop(i % nBins)
        bin.append(word)
        bins.insert(i % nBins, bin)

    for i in range(nBins):
        print(f'Bin {i} has size {len(bins[i])}')

    return bins



def bagOfWordsForSheetContinuous(bow: BagOfWords, sheet):
    bowsForSheet = bow.transform([sheet])
    nBins = bow.word_size // bow.window_step

    # Bow Transform creates a continous amount of words.
    # Each bin represents another offset and all Words within a bin create the continous signal
    wordsInBins = splitWordsIntoBins(bowsForSheet[0].split(' '), nBins)
    return wordsInBins
