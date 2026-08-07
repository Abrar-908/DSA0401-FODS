import math

mean = 200
sd = 350
n = 50

se = sd / math.sqrt(n)
t = mean / se

print("t Value =", t)

if t > 1.676:
    print("Reject Null Hypothesis")
    print("Model B performs better.")
else:
    print("Fail to Reject Null Hypothesis")