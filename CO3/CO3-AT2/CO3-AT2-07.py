import numpy as np

r = [4,5,3,4,5,4,3,5,4,4,5,3,4,5,4,3,5,4,5,4]

print("Mean =", np.mean(r))
print("Variance =", np.var(r))
print("Standard Error =", np.std(r) / np.sqrt(len(r)))