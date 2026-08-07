import numpy as np

e = [10,15,20,25,30,35,40,45,50,55]
b = [32,31,30,29,28,27,26,25,24,23]
bp = [150,148,145,142,138,134,130,126,122,118]

print("Mean =", np.mean(bp))
print("SD =", np.std(bp))
print("Cov(E,BP) =", np.cov(e, bp)[0][1])
print("Cov(BMI,BP) =", np.cov(b, bp)[0][1])
print("Corr(E,BP) =", np.corrcoef(e, bp)[0][1])
print("Corr(BMI,BP) =", np.corrcoef(b, bp)[0][1])