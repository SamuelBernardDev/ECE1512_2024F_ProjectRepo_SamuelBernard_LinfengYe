import matplotlib.pyplot as plt

Baseline = [92.63, 93.23, 94.99, 94.72, 97.16]
randominit = [95.17, 95.15, 95.30, 95.48, 97.23]
realinit = [95.26, 95.76, 95.98, 95.66, 97.16]

Classes = [10, 8, 6, 4, 2]

plt.plot(Classes, Baseline, label = "randomly selected images")
plt.plot(Classes, realinit, label = "DataDam with real initialization")
plt.plot(Classes, randominit, label = "DataDam with random initialization")
plt.xlabel("Number of classes")
plt.ylabel("Testing accuracy (%)")
plt.legend()
plt.show()