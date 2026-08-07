import numpy as np
from scipy.stats import ttest_rel

before = [50,52,49,55,53,51,54,50]
after = [58,60,57,62,59,61,63,58]

t, p = ttest_rel(after, before)

print("t-value =", t)
print("p-value =", p)

if p < 0.05:
    print("Reject Null Hypothesis")
    print("Marketing campaign increased sales.")
else:
    print("Fail to Reject Null Hypothesis")