import numpy as np

s = [72,75,68,80,77,73,79,81,76,74,78,69,70,82,71,75,77,73,80,76,74,79,72,81,78]

print("Mean =", np.mean(s))
print("Standard Deviation =", np.std(s))
print("Standard Error =", np.std(s) / np.sqrt(len(s)))