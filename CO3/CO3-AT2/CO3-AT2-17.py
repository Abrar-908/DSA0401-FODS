import math

n1 = 5000
x1 = 250

n2 = 5200
x2 = 312

p1 = x1 / n1
p2 = x2 / n2

p = (x1 + x2) / (n1 + n2)

se = math.sqrt(p * (1 - p) * ((1 / n1) + (1 / n2)))

z = (p2 - p1) / se

print("Conversion Rate A =", p1)
print("Conversion Rate B =", p2)
print("Z Value =", z)

if z > 1.645:
    print("Reject Null Hypothesis")
    print("Page B performs better.")
else:
    print("Fail to Reject Null Hypothesis")