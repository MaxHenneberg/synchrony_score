from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from dataplayground.AnomalyDetector import AnomalyDetector
from dataplayground.DataUtil import *

chunkSize = 250
autoencoder = AnomalyDetector(chunkSize)

ff_user_1, ff_user_2, ff_frame = extractUser('../resources/FF.xlsx')
ff_sync = calcSync(ff_user_1, ff_user_2)

ff_sync_splits = slidingWindow(ff_sync, chunkSize, 60)
print(ff_sync_splits.size)
train_data, test_data = train_test_split(ff_sync_splits, test_size=0.2, random_state=21)
train_data = normalizeData(train_data)
test_data = normalizeData(test_data)

autoencoder.compile(optimizer='adam', loss='mae')
history = autoencoder.fit(train_data, train_data,
                          epochs=60,
                          validation_data=(test_data, test_data),
                          shuffle=True)

reconstructions = autoencoder.predict(train_data)
train_loss = tf.keras.losses.mae(reconstructions, train_data)

threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)


def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)


normal = predict(autoencoder, test_data, threshold)

test_data_size = (int)(tf.size(test_data) / chunkSize)
fig, axs = plt.subplots(test_data_size)
fig.set_size_inches(10, 50)
fig.set_dpi(200)
fig.suptitle('Test')
plt.subplots_adjust(hspace=0.8)
print()
for i in range(test_data_size):
    max = (str)(round(tf.reduce_max(test_data[i]).numpy(),2))
    mean = (str)(round(tf.reduce_mean(test_data[i]).numpy(), 2))
    min = (str)(round(tf.reduce_min(test_data[i]).numpy(),2))
    axs[i].plot(np.arange(chunkSize), test_data[i], 'g' if normal[i] else 'r')
    axs[i].set_title('Min: '+min + '| Max: ' + max +'| Mean: '+mean)

plt.show()
