import numpy
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt

ff_raw = pd.read_excel('../resources/FF.xlsx')
ff = ff_raw.to_numpy()

frames_from = 4000
frames_to = 6000

ff_user_1 = ff[frames_from:frames_to, 3]
ff_user_1_norm = np.linalg.norm(ff_user_1)
ff_user_2 = ff[frames_from:frames_to, 4]
ff_user_2_norm = np.linalg.norm(ff_user_2)
ff_norm = np.fmax(ff_user_1_norm, ff_user_2_norm)
ff_user_1 = ff_user_1/ff_norm
ff_user_2 = ff_user_2/ff_norm
ff_frame = ff[frames_from:frames_to, 1]


ff_diff = np.abs(ff_user_1-ff_user_2)
ff_sum = ff_user_1+ff_user_2

ff_syc = ff_sum - ff_diff

# defines the figure to plot
plt.figure(figsize=(40, 6))
plt.xlabel('Frame')
plt.ylabel('Value')

# plot the original data
plt.plot(ff_frame, ff_user_1)
plt.plot(ff_frame, ff_user_2)
plt.plot(ff_frame, ff_syc)

plt.show()
