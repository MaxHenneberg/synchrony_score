from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from dataplayground.AnomalyDetectorRawData import AnomalyDetectorRawData, LayerDefinition
from dataplayground.DataUtil import *

chunkSize = 256
epochs = 60
layerDefinitions = [LayerDefinition(128, (1, 8), 'relu', 'same', (1, 4)),
                    LayerDefinition(2, (1, 4), 'relu', 'same', (1, 4)),
                    LayerDefinition(2, (1, 4), 'relu', 'same', (1, 4)),
                    LayerDefinition(128, (1, 8), 'relu', 'same', (1, 4))]
showLatentLayer = True

autoencoder = AnomalyDetectorRawData(chunkSize, layerDefinitions)

ff_user_1, ff_user_2, ff_frame = extractUser('../resources/FF.xlsx')

ff_user_1, ff_user_2 = findInterestingChunks(ff_user_1, ff_user_2, chunkSize)

ff_user_1 = normalizeData(ff_user_1)
ff_user_2 = normalizeData(ff_user_2)

input = shapeInput(ff_user_1, ff_user_2, 2)

train_data, test_data = train_test_split(input, test_size=0.2, random_state=21)

autoencoder.compile(optimizer='adam', loss='mae')
history = autoencoder.fit(train_data, train_data,
                          epochs=epochs,
                          validation_data=(test_data, test_data),
                          shuffle=True, verbose=False)

reconstructions = autoencoder.predict(train_data)

mae = []
for i in range(len(reconstructions)):
    user1Rec = np.squeeze(reconstructions[i][0])
    user1 = np.squeeze(train_data[i][0])
    user2Rec = np.squeeze(reconstructions[i][1])
    user2 = np.squeeze(train_data[i][1])
    mae1 = tf.keras.losses.mae(user1Rec, user1)
    mae2 = tf.keras.losses.mae(user2Rec, user2)
    mae.append((mae1 + mae2) / 2)

train_loss = np.array(mae)
threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)


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


# print(input.shape)
# latent = autoencoder.encoder(input)
# print(latent[0].shape)
#
# for sample in latent[0:5]:
#     fix, axs = plt.subplots(2)
#     axs[0].set_title('User: 1 Latent')
#     axs[0].plot(np.arange(16), sample[0])
#     axs[1].set_title('User: 2 Latent')
#     axs[1].plot(np.arange(16), sample[1])
#     plt.show()
#
# printStats(chunkSize, epochs, history, [], layerDefinitions)
saveWeights(autoencoder, 'autoencoderRAW')

loadedAutoencoder = AnomalyDetectorRawData(chunkSize, layerDefinitions)
loadWeights(loadedAutoencoder, 'autoencoderRAW')
normal = predict(loadedAutoencoder, test_data, threshold)
for i in range(len(test_data)):
    sample = test_data[i]
    latent = autoencoder.encoder(np.expand_dims(sample, axis=0))
    latentResult = latent[0]
    result = autoencoder.decoder(latent)
    sampleAfterCall = result[0]
    latentLength = calcLatentLength(chunkSize, layerDefinitions)
    # print(sampleAfterCall)

    if (showLatentLayer):
        fix, axs = plt.subplots(6)

        for j in range(len(axs)):
            axs[j].set_ylim([0, 1])

        axs[0].set_title('User: 1 Raw')
        axs[0].plot(np.arange(256), sample[0], 'g' if normal[i] else 'r')
        axs[1].set_title('User: 1 Latent')
        axs[1].plot(np.arange(latentLength), latentResult[0])
        axs[2].set_title('User: 1 Reconstructed')
        axs[2].plot(np.arange(256), sampleAfterCall[0])
        axs[3].set_title('User: 2 Raw')
        axs[3].plot(np.arange(256), sample[1], 'g' if normal[i] else 'r')
        axs[4].set_title('User: 1 Latent')
        axs[4].plot(np.arange(latentLength), latentResult[1])
        axs[5].set_title('User: 2 Reconstructed')
        axs[5].plot(np.arange(256), sampleAfterCall[1])
        plt.show()
    else:
        fix, axs = plt.subplots(4)

        for j in range(len(axs)):
            axs[j].set_ylim([0, 1])

        axs[0].set_title('User: 1 Raw')
        axs[0].plot(np.arange(256), sample[0], 'g' if normal[i] else 'r')
        axs[1].set_title('User: 1 Reconstructed')
        axs[1].plot(np.arange(256), sampleAfterCall[0])
        axs[2].set_title('User: 2 Raw')
        axs[2].plot(np.arange(256), sample[1], 'g' if normal[i] else 'r')
        axs[3].set_title('User: 2 Reconstructed')
        axs[3].plot(np.arange(256), sampleAfterCall[1])
        plt.show()


printStats(chunkSize, epochs, history, [], layerDefinitions)


