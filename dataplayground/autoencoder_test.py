import numpy as np
from sklearn.model_selection import train_test_split

from dataplayground.AnomalyDetectorRawData import AnomalyDetectorRawData
from dataplayground.DataUtil import extractUser, findInterestingChunks, normalizeData

chunkSize = 256
kernelWidth = 32
stride = 8
autoencoder = AnomalyDetectorRawData(chunkSize, kernelWidth, stride)

ff_user_1, ff_user_2, ff_frame = extractUser('../resources/FF.xlsx')

ff_user_1, ff_user_2 = findInterestingChunks(ff_user_1, ff_user_2, chunkSize)

# print(ff_user_1.shape)
# print(ff_user_2.shape)

ff_user_1 = normalizeData(ff_user_1)
ff_user_2 = normalizeData(ff_user_2)

def shapeInput(user1, user2):
    output = [np.expand_dims((ff_user_1[0], ff_user_2[0]), axis=2)]
    for i in range(len(ff_user_1) - 1):
        outStep = [np.expand_dims((ff_user_1[i + 1], ff_user_2[i + 1]), axis=2)]
        output = np.concatenate((output, outStep))
    return np.array(output)


input = shapeInput(ff_user_1, ff_user_2)
# input = np.reshape(input, (2, 256))
# print(input.shape)

# ff_sync = calcSync(ff_user_1, ff_user_2)
#
# ff_sync_splits = slidingWindowMultipleArrays(ff_sync, 50, 10)
# print(ff_sync_splits.shape)
#
train_data, test_data = train_test_split(input, test_size=0.2, random_state=21)
# print(train_data.shape)
# train_data = normalizeData(train_data)
# test_data = normalizeData(test_data)
#
autoencoder.compile(optimizer='adam', loss='mae')

enc = autoencoder.encoder(train_data)
print(enc.shape)

# dec = autoencoder.decoder(enc)
# print(dec.shape)

