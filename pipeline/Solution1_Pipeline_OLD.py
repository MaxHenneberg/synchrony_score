import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from autoencoders.ConvolutionalAutoEncoder import LayerDefinition, ConvolutionalAutoEncoder
from dataplayground import DataUtil
from sklearn.model_selection import train_test_split

chunkSize = 256
epochs = 60
layerDefinitions = [LayerDefinition(32, (8, 1), 'relu', 'same', (4, 1)),
                    LayerDefinition(2, (4, 1), 'relu', 'same', (2, 1)),
                    LayerDefinition(2, (4, 1), 'relu', 'same', (2, 1)),
                    LayerDefinition(32, (8, 1), 'relu', 'same', (4, 1))]
showLatentLayer = True

autoencoder = ConvolutionalAutoEncoder(chunkSize, layerDefinitions)

xl = pd.ExcelFile('..\\resources\\Prepared_Data\\Eyeblink_Raw.xlsx')


def parseSheet(sheet, columnIndexList):
    sheet = sheet.to_numpy()

    result = list()
    for i in columnIndexList:
        result.append(sheet[:, i])

    return result


def normalizeData(data):
    min_val = tf.reduce_min(data).numpy()
    max_val = tf.reduce_max(data).numpy()
    return (data - min_val) / (max_val - min_val)


def processSheet(sheet):
    normedUser1 = normalizeData(sheet[1])
    normedUser2 = normalizeData(sheet[2])
    user1 = list()
    user2 = list()

    for i in range(len(sheet[0])):
        user1.append([sheet[0][i], normedUser1[i]])
        user2.append([sheet[0][i], normedUser2[i]])

    return [user1, user2]


def batchInput(user, batchSize, offset):
    return DataUtil.slidingWindow(user, batchSize, offset)


sheets = list()
for sheetName in xl.sheet_names:
    print(f"Processing: {sheetName}")
    sheets.append(parseSheet(xl.parse(sheetName), [1, 2, 3]))

inputData = np.array(sheets, dtype=object)

formedInput = list()
for sheet in inputData:
    formedInput.append(processSheet(sheet))

# Sheet, User, Frame, ValueInFrame (Timestamp, value1, value2,...)

batchedInput = list()
for sheet in formedInput:
    user1 = batchInput(sheet[0], chunkSize, chunkSize)
    user2 = batchInput(sheet[1], chunkSize, chunkSize)
    batchedInput.append([user1, user2])

# (sheet/9, user/2, batches/171, entriesPerBatch/64, (timestamp, value)/2)
print(np.array(batchedInput).shape)

trainDataUser1List = list()
testDataUser1List = list()
trainDataUser2List = list()
testDataUser2List = list()
for sheet in batchedInput:
    trainData1, testData1 = train_test_split(sheet[0], test_size=0.2, random_state=21)
    trainDataUser1List.append(trainData1)
    testDataUser1List.append(testData1)
    trainData2, testData2 = train_test_split(sheet[1], test_size=0.2, random_state=21)
    trainDataUser2List.append(trainData2)
    testDataUser2List.append(testData2)


def mergeDataLists(dataList1, dataList2):
    combinedBatch = list()
    for i in range(len(dataList1)):
        batch1 = dataList1[i]
        batch2 = dataList2[i]
        if (len(batch1) != len(batch2)):
            print('DataLists not equally Long')
            return
        combinedBatchEntry = list()
        for j in range(len(batch1)):
            batchEntry1 = batch1[j]
            batchEntry2 = batch2[j]
            combinedBatchEntryEntry = list()
            for x in range(len(batchEntry1)):
                batchEntryEntry1 = batchEntry1[x]
                batchEntryEntry2 = batchEntry2[x]
                result = [[batchEntryEntry1[1]], [batchEntryEntry2[1]]]
                combinedBatchEntryEntry.append(result)
            combinedBatchEntry.append(combinedBatchEntryEntry)
        combinedBatch.append(combinedBatchEntry)
    return combinedBatch


combinedTrainData = mergeDataLists(trainDataUser1List, trainDataUser2List)
combinedTestData = mergeDataLists(testDataUser1List, testDataUser2List)

allTrainData = list()
for sheet in combinedTrainData:
    for entry in sheet:
        allTrainData.append(entry)

allTestData = list()
for sheet in combinedTestData:
    for entry in sheet:
        allTestData.append(entry)

def toPlot(data):
    plot = list()
    for j in range(len(data)):
        plot.append((data[j][0][0], data[j][1][0]))
    return plot

autoencoder.compile(optimizer='adam', loss='mae')
history = autoencoder.fit(allTrainData, allTrainData,
                          epochs=epochs,
                          validation_data=(allTestData, allTestData),
                          shuffle=True, verbose=True)

reconstruction = autoencoder.predict(allTestData)
fig, axs = plt.subplots(10)
for i in range(5):
    axs[2*i].set_ylim([0, 1])
    axs[2*i].plot(np.arange(256), toPlot(allTestData[i]))
    axs[(2*i)+1].set_ylim([0, 1])
    axs[(2*i)+1].plot(np.arange(256), toPlot(reconstruction[i]))
plt.show()