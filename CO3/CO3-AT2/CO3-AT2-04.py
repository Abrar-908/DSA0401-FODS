import numpy as np

t = [12000,13500,14500,16000,17500,19000,21000,22500,24000,26000]
r = [0,5,10,15,20,25,30,35,40,45]
s = [55,50,48,45,42,38,35,32,28,25]
a = [2,3,4,5,6,8,10,12,14,16]

print("Traffic Mean =", np.mean(t))
print("Traffic Variance =", np.var(t))
print("Traffic SD =", np.std(t))

print("Rainfall Mean =", np.mean(r))
print("Rainfall Variance =", np.var(r))
print("Rainfall SD =", np.std(r))

print("Speed Mean =", np.mean(s))
print("Speed Variance =", np.var(s))
print("Speed SD =", np.std(s))

print("Accident Mean =", np.mean(a))
print("Accident Variance =", np.var(a))
print("Accident SD =", np.std(a))

print("Covariance Matrix\n", np.cov([t,r,s,a]))

print("Correlation Matrix\n", np.corrcoef([t,r,s,a]))