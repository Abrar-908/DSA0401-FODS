import math
print("===== Z-Statistics =====")

n1 = int(input("Enter sample size of Medicine A: "))
x1 = float(input("Enter mean reduction of Medicine A: "))
u1 = float(input("Enter Population Mean of Medicine A: "))
s1 = float(input("Enter standard deviation of Medicine A: "))

n2 = int(input("\nEnter sample size of Medicine B: "))
x2 = float(input("Enter mean reduction of Medicine B: "))
u2 = float(input("Enter Population Mean of Medicine B: "))
s2 = float(input("Enter standard deviation of Medicine B: "))
alpha = int(input("\nEnter significance level (1 or 5): "))

z = (x1 - x2)-(u1 - u2) / math.sqrt((s1**2 / n1) + (s2**2 / n2))

print("\nCalculated Z-value =", round(z, 4))

if alpha == 5:
    critical = 1.96
elif alpha == 1:
    critical = 2.576
else:
    print("Invalid significance level! Please enter 1 or 5.")
    exit()

print("Critical Z-value = ±", critical)

if abs(z) > critical:
    print("\nDecision: Reject the Null Hypothesis (H0)")
    print("Conclusion: There is a significant difference between the two medications.")
else:
    print("\nDecision: Fail to Reject the Null Hypothesis (H0)")
    print("Conclusion: There is no significant difference between the two medications.")