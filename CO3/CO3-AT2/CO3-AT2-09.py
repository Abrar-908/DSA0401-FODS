import math

mean = 498
sd = 12
n = 50
z = 1.96

se = sd / math.sqrt(n)
me = z * se

lower = mean - me
upper = mean + me

print("Standard Error =", se)
print("95% Confidence Interval = (", lower, ",", upper, ")")

if lower <= 500 <= upper:
    print("500 g is plausible.")
else:
    print("500 g is not plausible.")