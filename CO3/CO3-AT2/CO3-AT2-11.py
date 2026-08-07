import math

mean = 9.5
claimed = 10
sd = 1.8
n = 36

z = (mean - claimed) / (sd / math.sqrt(n))

print("Z Value =", z)

if abs(z) > 1.96:
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")