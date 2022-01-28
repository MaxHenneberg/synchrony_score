import matplotlib.pyplot as plt
import numpy as np
from pyts.bag_of_words import BagOfWords

from dataplayground import DataUtil
from dataplayground.BagOfWords import bagOfWordsForSheet, printBow
from utils.prepare_data import collectUserData

windowSize = 8
wordSize = 8
batchSize = 256
folder = 'BOW_INVERS'
user1, user2 = collectUserData("Eyeblink_Raw", lambda sheet: sheet["User1Blink"], lambda sheet: sheet["User2Blink"])
bow = BagOfWords(window_size=windowSize, word_size=wordSize,
                 window_step=windowSize, numerosity_reduction=False)

bowMapUser2 = dict()
bowListListUser2 = list()
maxColor = 0
for i, sheet in enumerate(user2):
    maxColor, bowList = bagOfWordsForSheet(bow, sheet, 256, 2, i, bowMapUser2, maxColor, folder)
    bowListListUser2.append(bowList)

bowMapUser1 = dict()
bowListListUser1 = list()
maxColor = 0
for i, sheet in enumerate(user1):
    maxColor, bowList = bagOfWordsForSheet(bow, sheet, 256, 1, i, bowMapUser1, maxColor, folder)
    bowListListUser1.append(bowList)

# Merge User 1 and User 2 Maps to Value Objects in a List
mergedList = list()
for word in bowMapUser1.keys():
    user1 = bowMapUser1.get(word)
    user2 = bowMapUser2.get(word)
    mergedVal = {'word': word, 'user1': user1['count'], 'user2': 0}
    if user2 is not None:
        mergedVal['user2'] = user2['count']
        bowMapUser2.pop(word)

    mergedList.append(mergedVal)

# Sort them by Count of User1
sortedList = sorted(mergedList, key=lambda item: item['user1'], reverse=True)
words = [i['word'] for i in sortedList]
valuesUser1 = [i['user1'] for i in sortedList]
valuesUser2 = [i['user2'] for i in sortedList]
mergedMap = bowMapUser1.copy()
for key in bowMapUser2.keys():
    if(mergedMap.get(key) is not None):
        mergedMap[key] = bowMapUser2.get(key)
print(
    f'Size Merged: {len(mergedMap.keys())}; Size User1 {len(bowMapUser1.keys())}; Size User 2 {len(bowMapUser2.keys())}')

# Print Bar Chart of Word Occurrences comparing User 1 and User 2
amtOfValues = 15
plt.clf()
X = np.arange(amtOfValues)
width = 0.35
fig = plt.figure()
fig.set_size_inches(20, 10)
fig.suptitle(
    f'Most common {amtOfValues} Words out of {len(mergedMap.keys())} (User1:{len(bowMapUser1.keys())}/User2:{len(bowMapUser2.keys())})')
plt.bar(X, valuesUser1[0:amtOfValues], width, label='User1')

plt.bar(X+width, valuesUser2[0:amtOfValues], width, label='User2')

for i in range(amtOfValues):
    plt.text(X[i], valuesUser1[i] + 5, valuesUser1[i])
    plt.text(X[i]+width, valuesUser2[i] + 5, valuesUser2[i])
plt.xticks(X+width/2, words[0:amtOfValues])
plt.legend(loc='best')
plt.tight_layout()
DataUtil.savePlot(plt, folder, f'BOW_Summary')

# Plot Linecharts representing the samples of the most common words.
fig, axs = plt.subplots(amtOfValues)
fig.set_size_inches(20, 40)
for i in range(amtOfValues):
    word = words[i]
    count = bowMapUser1.get(word)['count']
    sample = bowMapUser1.get(word)['sample'] / count
    axs[i].set_ylim([-0.1, 1])
    axs[i].plot(np.arange(wordSize), sample)
    for j in range(wordSize):
        axs[i].text(j, sample[0:wordSize][j] + 0.1, sample[0:wordSize][j])
    axs[i].set_title(label=word)
plt.tight_layout()
DataUtil.savePlot(plt, folder, f'Word_Samples')

# Show User1 User2 side by Side

mergedSheets = list()
for sheetUser1, sheetUser2 in zip(bowListListUser1, bowListListUser2):
    mergedBatches = list()
    for batchUser1, batchUser2 in zip(sheetUser1, sheetUser2):
        mergedBatches.append(batchUser1)
        mergedBatches.append(batchUser2)
    mergedSheets.append(mergedBatches)



