import numpy as np
from pyts.bag_of_words import BagOfWords

from SynchronyScore.SyncScore import calculateSyncScoreForTimeSeries

windowSize = 8
wordSize = 8
nBins = 2
windowStep = 1
bow = BagOfWords(window_size=windowSize, word_size=wordSize,
                 window_step=windowStep, numerosity_reduction=False, n_bins=nBins)


def derivative(excel):
    dx = 0.1
    derivedSheets = list()
    for sheet in excel:
        derivedSheets.append(np.gradient(sheet, dx))
    return derivedSheets


calculateSyncScoreForTimeSeries(bow, "Smile", lambda sheet: sheet["User1Smile"],
                                lambda sheet: sheet["User2Smile"], derivative, derivative,
                                [[-0.1, 1.5], [-1.2, 1.2], [-1.2, 1.2]], "D_SMILE_Derivative")
