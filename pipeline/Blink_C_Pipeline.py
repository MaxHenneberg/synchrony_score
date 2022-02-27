from pyts.bag_of_words import BagOfWords

from SynchronyScore.SyncScore import calculateSyncScoreForTimeSeries

windowSize = 8
wordSize = 8
nBins = 2
windowStep = 1
bow = BagOfWords(window_size=windowSize, word_size=wordSize,
                 window_step=windowStep, numerosity_reduction=False, n_bins=nBins)

calculateSyncScoreForTimeSeries(bow, "Eyeblink_Classified", lambda sheet: sheet["User1Blink"],
                                lambda sheet: sheet["User2Blink"], lambda data: data, lambda data: data, [[-0.1, 1.5], [-0.1, 1.5], [-0.1, 1.5]],"Blink_C")
