import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\Downloads\exp 26 FDS.csv")

print("\nCLINICAL TRIAL DATA\n")
print(df)

# Separate the two groups
control = df[df["Group"] == "Control"]["Outcome_Score"]
treatment = df[df["Group"] == "Treatment"]["Outcome_Score"]

# Descriptive statistics
print("\nDESCRIPTIVE STATISTICS")
print("Control Mean:", control.mean())
print("Treatment Mean:", treatment.mean())
print("Control Standard Deviation:", control.std())
print("Treatment Standard Deviation:", treatment.std())

# Hypothesis
# H0: There is no significant difference between the two groups.
# H1: There is a significant difference between the two groups.

# Independent two-sample t-test
t_stat, p_value = stats.ttest_ind(
    treatment,
    control,
    equal_var=False
)

print("\nHYPOTHESIS TESTING")
print("T-statistic:", t_stat)
print("P-value:", p_value)

# Significance level
alpha = 0.05

if p_value < alpha:
    print("\nDecision: Reject the Null Hypothesis (H0)")
    print("Conclusion: The treatment has a statistically significant effect.")
else:
    print("\nDecision: Fail to Reject the Null Hypothesis (H0)")
    print("Conclusion: The treatment does not have a statistically significant effect.")

# Visualization
plt.figure(figsize=(10, 6))

plt.boxplot(
    [control, treatment],
    labels=["Control", "Treatment"],
    patch_artist=True
)

plt.ylabel("Outcome Score")
plt.title("Clinical Trial: Control vs Treatment")

# Display p-value on graph
y_max = df["Outcome_Score"].max()

plt.text(
    1.5,
    y_max + 1,
    f"p-value = {p_value:.6f}",
    ha="center",
    fontsize=12
)

plt.text(
    1.5,
    y_max - 1,
    "Statistically Significant" if p_value < alpha
    else "Not Statistically Significant",
    ha="center",
    fontsize=11
)

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()