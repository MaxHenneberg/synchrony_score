from pyts.bag_of_words import BagOfWords

from SynchronyScore.SyncScore import calculateSyncScoreForTimeSeries

windowSize = 8
wordSize = 8
nBins = 3
windowStep = 1
bow = BagOfWords(window_size=windowSize, word_size=wordSize,
                 window_step=windowStep, numerosity_reduction=False, n_bins=nBins)

calculateSyncScoreForTimeSeries(bow, "Smile", lambda sheet: sheet["User1Smile"],
                                lambda sheet: sheet["User2Smile"], lambda data: data, lambda data: data,[[-0.1, 1.5], [-0.1, 1.5], [-0.1, 1.5]], "D_SMILE_R_CLOSSNESS_NO_BOW")
