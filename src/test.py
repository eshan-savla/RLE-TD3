import matplotlib.pyplot as plt    
import numpy as np

x = np.arange(0, 5, 0.0000005)
y = np.add(-np.exp(x), np.exp(4.9999995))
print(np.max(y), np.exp(max(x)))
y = y/np.max(y)
plt.plot(x,y)
plt.show()