import math

n1 = 100000
n2 = 100000

p1 = 0.048
p2 = 0.053

x1 = n1 * p1
x2 = n2 * p2

p = (x1 + x2) / (n1 + n2)

se = math.sqrt(p * (1 - p) * ((1/n1) + (1/n2)))
z = (p2 - p1) / se

print("Z Value =", z)

if z > 1.645:
    print("Reject Null Hypothesis")
    print("New Recommendation System is better.")
else:
    print("Fail to Reject Null Hypothesis")