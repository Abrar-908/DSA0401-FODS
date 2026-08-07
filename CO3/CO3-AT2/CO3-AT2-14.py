import math

n = 500
d = 22
z = 1.96

p = d / n
se = math.sqrt((p * (1 - p)) / n)
me = z * se

lower = p - me
upper = p + me

print("Defect Rate =", p)
print("95% Confidence Interval = (", lower, ",", upper, ")")