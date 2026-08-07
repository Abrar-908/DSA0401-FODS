a = 4.3
b = 4.0

a_ci = (4.1, 4.5)
b_ci = (3.8, 4.2)

print("Branch A CI =", a_ci)
print("Branch B CI =", b_ci)

if a_ci[0] <= b_ci[1] and b_ci[0] <= a_ci[1]:
    print("Confidence intervals overlap.")
    print("Cannot conclude Branch A performs better.")
else:
    print("Confidence intervals do not overlap.")
    print("Branch A performs better.")