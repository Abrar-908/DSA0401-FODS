import numpy as np

v = [5,8,10,12,15,18,20,23,25,28]
t = [20,25,30,35,40,45,50,55,60,70]
p = [1200,1800,2500,3200,4000,5000,6200,7300,8500,10000]

print("Mean =", np.mean(p))
print("Variance =", np.var(p, ddof=1))
print("SD =", np.std(p, ddof=1))

print("Cov(Visits,Purchase) =", np.cov(v,p)[0][1])
print("Cov(Time,Purchase) =", np.cov(t,p)[0][1])

print("Corr(Visits,Purchase) =", np.corrcoef(v,p)[0][1])
print("Corr(Time,Purchase) =", np.corrcoef(t,p)[0][1])