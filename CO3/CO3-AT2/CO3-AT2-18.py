import math

n1 = 400
m1 = 18
sd1 = 5

n2 = 420
m2 = 20
sd2 = 6

se = math.sqrt((sd1**2/n1) + (sd2**2/n2))
z = (m2 - m1) / se

print("Z Value =", z)

if z > 1.645:
    print("Reject Null Hypothesis")
    print("Feature B increases engagement.")
else:
    print("Fail to Reject Null Hypothesis")