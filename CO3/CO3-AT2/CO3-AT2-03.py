import numpy as np

t = [5,8,10,12,15,18,20,22,25,28]
p = [2,3,4,5,6,7,8,9,10,11]
r = [60,65,70,74,79,84,88,91,95,98]

print("Mean =", np.mean(r))
print("Variance =", np.var(r))
print("SD =", np.std(r))

print("Cov(Training,Rating) =", np.cov(t, r)[0][1])
print("Cov(Projects,Rating) =", np.cov(p, r)[0][1])

print("Corr(Training,Rating) =", np.corrcoef(t, r)[0][1])
print("Corr(Projects,Rating) =", np.corrcoef(p, r)[0][1])