import pandas as pd
import numpy as np
from scipy import stats

data = pd.read_csv("ab_test.csv")

design_A = data[data["design"] == "A"]["conversion_rate"].dropna().values
design_B = data[data["design"] == "B"]["conversion_rate"].dropna().values

mean_A = np.mean(design_A)
mean_B = np.mean(design_B)

t_stat, p_value = stats.ttest_ind(
    design_A,
    design_B,
    equal_var=False
)

print("Mean Conversion Rate - Design A:", mean_A)
print("Mean Conversion Rate - Design B:", mean_B)
print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("There is a statistically significant difference between Design A and Design B.")
else:
    print("There is no statistically significant difference between Design A and Design B.")