from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pyts.bag_of_words import BagOfWords

from SynchronyScore.SyncScoreUtils import calcSyncScore, createWordMap, mergeWordMapsForStoring
from dataplayground import DataUtil
from SynchronyScore.BagOfWords import bagOfWordsForSheetContinuous
from dataplayground.DataUtil import normalizeData
from utils.prepare_data import collectUserData

runId = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')

windowSize = 8
wordSize = 8
nBins = 2
windowStep = 1
batchSize = 256
folder = 'BOW_Continuous_Merged'
user1, user2 = collectUserData("Eyeblink_CLassified", lambda sheet: sheet["User1Blink"],
                               lambda sheet: sheet["User2Blink"])

user1 = [normalizeData(sheet) for sheet in user1]
user2 = [normalizeData(sheet) for sheet in user2]

bow = BagOfWords(window_size=windowSize, word_size=wordSize,
                 window_step=windowStep, numerosity_reduction=False, n_bins=nBins)

summedWordBinsUser1 = []
for i, sheet in enumerate(user1):
    print(f'Processed Sheet {i} for User 1')
    wordBinsForSheet = bagOfWordsForSheetContinuous(bow, sheet)
    summedWordBinsUser1.append(wordBinsForSheet)
    print(f'Finished Sheet {i} for User 1')

summedWordBinsUser2 = []
for i, sheet in enumerate(user2):
    print(f'Processed Sheet {i} for User 2')
    wordBinsForSheet = bagOfWordsForSheetContinuous(bow, sheet)
    summedWordBinsUser2.append(wordBinsForSheet)
    print(f'Finished Sheet {i} for User 2')

mergedSheets = list()
for i, (sheetU1, sheetU2) in enumerate(zip(summedWordBinsUser1, summedWordBinsUser2)):
    wordMapUser1 = createWordMap(sheetU1[0])
    wordMapUser2 = createWordMap(sheetU2[0])
    mergedSheets.append(mergeWordMapsForStoring(wordMapUser1, wordMapUser2))

DataUtil.saveData(["word", "user1Count", "user2Count"], mergedSheets, folder, f'Words-{nBins}-Bins', runId)

syncScoreU1U2 = calcSyncScore(summedWordBinsUser1, user1, summedWordBinsUser2, user2, wordSize)
syncScoreU1U2WithTimeStamp = [[np.arange(len(score)), score] for score in syncScoreU1U2]
syncScoreU2U1 = calcSyncScore(summedWordBinsUser2, user2, summedWordBinsUser1, user1, wordSize)
syncScoreU2U1WithTimeStamp = [[np.arange(len(score)), score] for score in syncScoreU2U1]

DataUtil.saveData(["timestamp", "syncScore"], syncScoreU1U2WithTimeStamp, folder, f'SyncScoreU1U2-{nBins}-Bins', runId)
DataUtil.saveData(["timestamp", "syncScore"], syncScoreU2U1WithTimeStamp, folder, f'SyncScoreU2U1-{nBins}-Bins', runId)

syncScore = list()
for s1, s2 in zip(syncScoreU1U2, syncScoreU2U1):
    syncScore.append((np.array(s1) + np.array(s2)) / 2)

syncScoreWithTimeStamp = [[np.arange(len(score)), score] for score in syncScore]

DataUtil.saveData(["timestamp", "syncScore"], syncScoreWithTimeStamp, folder, f'SyncScore-{nBins}-Bins', runId)

def minMaxColor(value):
    return max(0, min(1, value))


def calcColor(scoreForWord):
    return [minMaxColor(1 - scoreForWord), minMaxColor(scoreForWord), 0]


def wordColor(word):
    r = 0
    b = 0
    for i, c in enumerate(word):
        if c == 'a':
            r += (i*i)
        else:
            b += (i*i)

    return [r/255, 0, b/255]


for s, (sheet, (sheetWordsUser1, sheetWordsUser2)) in enumerate(zip(syncScore, zip(summedWordBinsUser1, summedWordBinsUser2))):
    amtOfPlots = len(sheet) // batchSize
    wordsPerBatch = batchSize // wordSize
    fig = plt.figure()
    fig.set_size_inches(20, 100)
    gs0 = fig.add_gridspec(amtOfPlots, 1)
    for i in range(amtOfPlots):
        subplot = gs0[i].subgridspec(3, 1, hspace=0)
        start = (i * batchSize)
        end = min(((i + 1) * batchSize) + 1, len(sheet)-1)
        ax0 = fig.add_subplot(subplot[0, 0])
        plt.setp(ax0.get_xticklabels(), visible=False)
        ax0.set_ylim([-0.1, 1.5])
        for j in range(wordsPerBatch):
            startWord = (j * wordSize)
            endWord = min(((j + 1) * wordSize) + 1, batchSize - 1)
            scoreForWord = sheet[start:end][startWord]
            if scoreForWord > 0:
                ax0.text(start + startWord, scoreForWord + 0.1, round(scoreForWord, 2), fontsize=8)
            ax0.plot(np.arange(start + startWord, start + endWord), sheet[start:end][startWord:endWord],
                     c=calcColor(scoreForWord))
        ax1 = fig.add_subplot(subplot[1, 0], sharex=ax0, sharey=ax0)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_ylim([-0.1, 1.5])
        ax1.plot(np.arange(start, end - 1), user1[s][start:end-1])
        startOfWords = (i * wordsPerBatch)
        endOfWords = (i * wordsPerBatch) + wordsPerBatch
        ax2 = fig.add_subplot(subplot[2, 0], sharex=ax0, sharey=ax0)
        ax2.set_ylim([-0.1, 1.5])
        ax2.plot(np.arange(start, end - 1), user2[s][start:end-1])
        for j in range(wordsPerBatch):
            word1 = sheetWordsUser2[0][startOfWords:endOfWords][j]
            word1Color = wordColor(word1)
            word2 = sheetWordsUser1[0][startOfWords:endOfWords][j]
            word2Color = wordColor(word2)
            for cIdx, (c1, c2) in enumerate(zip(word1, word2)):
                posInPlot = min(start + (j * wordSize) + cIdx, len(user1[s]) - 1)
                ax1.text(posInPlot, user1[s][posInPlot] + 0.2, c1, color=word1Color)
                ax2.text(posInPlot, user2[s][posInPlot] + 0.2, c2, color=word2Color)

    plt.tight_layout()
    DataUtil.savePlot(plt, folder, f'Sheet {s}-{nBins}-Bins', runId)

