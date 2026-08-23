import pandas as pd
import numpy as np
from scipy import stats

data = pd.read_csv("blood_pressure.csv")

drug = data[data["group"] == "Drug"]["reduction"].dropna().values
placebo = data[data["group"] == "Placebo"]["reduction"].dropna().values

def confidence_interval(data):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    t_value = stats.t.ppf(0.975, n - 1)
    margin = t_value * se
    return mean, mean - margin, mean + margin

drug_mean, drug_lower, drug_upper = confidence_interval(drug)
placebo_mean, placebo_lower, placebo_upper = confidence_interval(placebo)

print("New Drug Group")
print("Mean Reduction:", drug_mean)
print("95% Confidence Interval:", drug_lower, "to", drug_upper)

print("\nPlacebo Group")
print("Mean Reduction:", placebo_mean)
print("95% Confidence Interval:", placebo_lower, "to", placebo_upper)