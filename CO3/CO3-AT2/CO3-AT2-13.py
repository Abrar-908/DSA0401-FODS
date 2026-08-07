import math

mean = 4.8
sd = 1.5
n = 100
z = 1.96

se = sd / math.sqrt(n)
me = z * se

lower = mean - me
upper = mean + me

print("Estimated Population Mean =", mean)
print("Standard Error =", se)
print("95% Confidence Interval = (", lower, ",", upper, ")")