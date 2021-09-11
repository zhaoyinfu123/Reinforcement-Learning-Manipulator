import numpy as np
import matplotlib.pyplot as plt


dist = np.linspace(0, 1, 100)
r = (np.exp(-dist*7)-1)*10
plt.plot(dist, r)
plt.show()
