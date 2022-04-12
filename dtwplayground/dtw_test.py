import numpy as np
from matplotlib import pyplot as plt

from dataplayground.DataUtil import normalizeData
from utils.prepare_data import collectUserData
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

srcExcel = "Improved_Duchenne_Smile"
user1GetColumn = lambda sheet: sheet["User1Smile"]
user2GetColumn = lambda sheet: sheet["User2Smile"]
user1PreProcessing = lambda data: data
user2PreProcessing = lambda data: data

user1, user2 = collectUserData(srcExcel, user1GetColumn,
                               user2GetColumn)

# Normalize Time Series
user1 = [normalizeData(sheet) for sheet in user1]
user2 = [normalizeData(sheet) for sheet in user2]

# Apply Custom Preprocessing to normalized UserData
user1 = user1PreProcessing(user1)
user2 = user2PreProcessing(user2)

s1 = np.array([0, 1, 0], dtype=np.double)
s2 = np.array([0, -1, 0], dtype=np.double)
smerged = np.vstack([s1, s2])
paths = dtw.distance_matrix(smerged)
paths = paths / 2
diagonal = np.diagonal(paths)
# best_path = dtwplayground.best_path(paths)
# print(len(best_path))
# dtwvis.plot_warpingpaths(s1, s2, paths, best_path, filename="path.png", showlegend=True)

fig, axs = plt.subplots(5, 1, constrained_layout=True)
# fig.set_size_inches(20, 2 * amtOfPlots)
axs.flat[0].plot(np.arange(len(s1)), s1[:len(s1)])
axs.flat[1].plot(np.arange(len(s1)), s2[:len(s1)])
axs.flat[2].plot(np.arange(len(s1)), diagonal[:len(s1)])
# print(len(axs.flat[3:5]))
# dtwvis.plot_warping(s1, s2, best_path, axs=axs.flat[3:5], fig=fig)
plt.show()
