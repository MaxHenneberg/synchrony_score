import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from dataplayground.AnomalyDetectorRawData import LayerDefinition, AnomalyDetectorRawData
from dataplayground.DataUtil import *

chunkSizeLayer1 = 256
chunkSizeLayer2 = 16
epochs = 60
layerDefinitionsLevel1 = [LayerDefinition(128, (1, 8), 'relu', 'same', (1, 4)),
                          LayerDefinition(2, (1, 4), 'relu', 'same', (1, 4)),
                          LayerDefinition(2, (1, 4), 'relu', 'same', (1, 4)),
                          LayerDefinition(128, (1, 8), 'relu', 'same', (1, 4))]
layerDefinitionsLevel2 = [LayerDefinition(16, (1, 4), 'relu', 'same', (1, 2)),
                          LayerDefinition(2, (1, 2), 'relu', 'same', (1, 2)),
                          LayerDefinition(2, (1, 2), 'relu', 'same', (1, 2)),
                          LayerDefinition(16, (1, 4), 'relu', 'same', (1, 2))]
showLatentLayer = True

firstLevelAutoEncoder = AnomalyDetectorRawData(chunkSizeLayer1, layerDefinitionsLevel1)
loadWeights(firstLevelAutoEncoder, 'autoencoderRAW')

ff_user_1, ff_user_2, ff_frame = extractUser('../resources/FF.xlsx')

ff_user_1, ff_user_2 = findInterestingChunks(ff_user_1, ff_user_2, chunkSizeLayer1)

ff_user_1 = normalizeData(ff_user_1)
ff_user_2 = normalizeData(ff_user_2)

# (31, 2, 256, 1)
input = shapeInput(ff_user_1, ff_user_2, 2)

# (31, 2, 16, 2)
latent = firstLevelAutoEncoder.encoder(input)


def reshapeFirstLevel(input):
    """
    Split Input of 2 Users into 2 seperate Arrays
    :param input: (31,2,16,2)   (sample, user, sampleLength, LatentFilterSize)
    :return: output (31,2,16), (31,2,16)
    """
    print(input.shape)
    user1Sample = np.array([])
    user2Sample = np.array([])
    for i in range((len(input))):
        # (16,2)
        user1 = input[i][0]
        # (16,2)
        user2 = input[i][1]

        if (len(user1Sample) == 0):
            user1Sample = np.array([np.transpose(user1)])
            user2Sample = np.array([np.transpose(user2)])
        else:
            user1Sample = np.concatenate((user1Sample, [np.transpose(user1)]))
            user2Sample = np.concatenate((user2Sample, [np.transpose(user2)]))

    return user1Sample, user2Sample


user1, user2 = reshapeFirstLevel(latent)

train_data1, test_data1 = train_test_split(user1, test_size=0.2, random_state=21)
train_data2, test_data2 = train_test_split(user2, test_size=0.2, random_state=21)

train_data1 = np.expand_dims(train_data1, axis=3)
test_data1 = np.expand_dims(test_data1, axis=3)
train_data2 = np.expand_dims(train_data2, axis=3)
test_data2 = np.expand_dims(test_data2, axis=3)

secondLevelAutoEncoder = AnomalyDetectorRawData(chunkSizeLayer2, layerDefinitionsLevel2)

secondLevelAutoEncoder.compile(optimizer='adam', loss='mae')
history = secondLevelAutoEncoder.fit(train_data1, train_data2,
                                     epochs=epochs,
                                     validation_data=(test_data1, test_data2),
                                     shuffle=True, verbose=True)


def predict(model, data, threshold):
    reconstructions = model(data)
    result = []
    for i in range(len(reconstructions)):
        user1 = np.squeeze(data[i][0])
        user1Rec = np.squeeze(reconstructions[i][0])
        user2 = np.squeeze(data[i][1])
        user2Rec = np.squeeze(reconstructions[i][1])
        loss1 = tf.keras.losses.mae(user1Rec, user1)
        loss2 = tf.keras.losses.mae(user2Rec, user2)
        result.append(tf.math.less(loss1, threshold) & tf.math.less(loss2, threshold))
    return np.array(result)


reconstructions = secondLevelAutoEncoder.predict(train_data1)

mae = []
for i in range(len(reconstructions)):
    user1Rec = np.squeeze(reconstructions[i][0])
    user1 = np.squeeze(train_data1[i][0])
    user2Rec = np.squeeze(reconstructions[i][1])
    user2 = np.squeeze(train_data1[i][1])
    mae1 = tf.keras.losses.mae(user1Rec, user1)
    mae2 = tf.keras.losses.mae(user2Rec, user2)
    mae.append((mae1 + mae2) / 2)

train_loss = np.array(mae)
threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)

normal = predict(secondLevelAutoEncoder, test_data1, threshold)
for i in range(len(test_data1)):
    user1 = test_data1[i]
    user2 = test_data2[i]
    latent = secondLevelAutoEncoder.encoder(np.expand_dims(user1, axis=0))
    latentResult = latent[0]
    result = secondLevelAutoEncoder.decoder(latent)
    sampleAfterCall = result[0]
    latentLength = calcLatentLength(chunkSizeLayer2, layerDefinitionsLevel2)
    # print(sampleAfterCall)

    fix, axs = plt.subplots(4)

    for j in range(len(axs)):
        axs[j].set_ylim([0, 1])

    axs[0].set_title('User: 1 Latent')
    axs[0].plot(np.arange(chunkSizeLayer2),np.transpose(np.squeeze(user1, axis=2)))
    axs[1].set_title('Latent')
    axs[1].plot(np.arange(latentLength), latentResult[0])
    axs[2].set_title('User: 2 Latent Rec')
    axs[2].plot(np.arange(chunkSizeLayer2), np.transpose(np.squeeze(sampleAfterCall, axis=2)))
    axs[3].set_title('User: 2 Latent')
    axs[3].plot(np.arange(chunkSizeLayer2), np.transpose(np.squeeze(user2, axis=2)))
    plt.show()

printStats(chunkSizeLayer2, epochs, history, [], layerDefinitionsLevel2)
