import numpy as np
from pyts.bag_of_words import BagOfWords

from SynchronyScore.SyncScore import calculateSyncScoreForTimeSeries

windowSize = 4
wordSize = 4
nBins = 3
windowStep = 1
bow = BagOfWords(window_size=windowSize, word_size=wordSize,
                 window_step=windowStep, numerosity_reduction=False, n_bins=nBins)


def helper(val, maxVal, minVal):
    if val > maxVal:
        return 1
    else:
        if val < minVal:
            return -1
        else:
            return 0

def absDerivative(excel):
    dx = 0.1
    derivedSheets = list()
    for sheet in excel:
        der = np.gradient(sheet, dx)
        min_val = min(der) * 0.3
        max_val = max(der) * 0.3
        absDer = [helper(val, max_val, min_val) for val in der]
        derivedSheets.append(absDer)
    return derivedSheets


calculateSyncScoreForTimeSeries(bow, "Smile", lambda sheet: sheet["User1Smile"],
                                lambda sheet: sheet["User2Smile"], absDerivative, absDerivative,
                                [[-0.1, 1.5], [-1.1, 1.1], [-1.1, 1.1]], "D_SMILE_Abs_Derivative")
