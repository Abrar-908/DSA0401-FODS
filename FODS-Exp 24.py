import pandas as pd
import numpy as np
from scipy import stats

data = pd.read_csv("rare_elements.csv")

values = data["concentration"].dropna().values

sample_size = int(input("Enter sample size: "))
confidence_level = float(input("Enter confidence level (e.g., 0.95): "))
precision = float(input("Enter desired level of precision: "))

if sample_size > len(values):
    print("Sample size is greater than the available data.")
else:
    sample = np.random.choice(values, sample_size, replace=False)

    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)

    alpha = 1 - confidence_level

    t_value = stats.t.ppf(
        1 - alpha / 2,
        sample_size - 1
    )

    standard_error = sample_std / np.sqrt(sample_size)

    margin_of_error = t_value * standard_error

    lower_limit = sample_mean - margin_of_error
    upper_limit = sample_mean + margin_of_error

    print("\nPoint Estimate:")
    print("Sample Mean =", sample_mean)

    print("\nConfidence Interval:")
    print("Lower Limit =", lower_limit)
    print("Upper Limit =", upper_limit)

    print("\nMargin of Error =", margin_of_error)

    if margin_of_error <= precision:
        print("Desired level of precision is achieved.")
    else:
        print("Desired level of precision is not achieved.")
        print("Increase the sample size for better precision.")