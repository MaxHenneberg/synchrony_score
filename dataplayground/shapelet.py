import numpy as np
from matplotlib import pyplot as plt
from pyts.transformation import ShapeletTransform
X = [[0, 2, 3, 4, 3, 2, 1],
     [0, 1, 3, 4, 3, 4, 5],
     [2, 1, 0, 2, 1, 5, 4],
     [1, 2, 2, 1, 0, 3, 5]]
y = [0, 0, 1, 1]
#
# for i in range(len(X)):
#     plt.plot(np.arange(7), X[i])
#
# plt.show()

st = ShapeletTransform(n_shapelets=2)
X_new = st.fit_transform(X, y)
# indices = [idx, start, end]
print(st.indices_)
for i, index in enumerate(st.indices_):
    idx, start, end = index
    plt.plot(X[idx], color='C{}'.format(i),
             label='Sample {}'.format(idx))
    plt.plot(np.arange(start, end), X[idx][start:end],
             lw=5, color='C{}'.format(i))
plt.show()
