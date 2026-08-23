import pandas as pd
import numpy as np
from scipy import stats

# Read CSV file
data = pd.read_csv("customer_reviews.csv")

# Extract ratings
ratings = data["rating"].dropna().values

# Calculate sample size
n = len(ratings)

# Calculate sample mean
mean_rating = np.mean(ratings)

# Calculate sample standard deviation
std_rating = np.std(ratings, ddof=1)

# Confidence level
confidence_level = 0.95
alpha = 1 - confidence_level

# Calculate t-value
t_value = stats.t.ppf(1 - alpha / 2, n - 1)

# Calculate standard error
standard_error = std_rating / np.sqrt(n)

# Calculate margin of error
margin_of_error = t_value * standard_error

# Calculate confidence interval
lower_limit = mean_rating - margin_of_error
upper_limit = mean_rating + margin_of_error

# Display results
print("Customer Review Analysis")
print("-------------------------")
print("Number of Reviews:", n)
print("Average Rating:", mean_rating)
print("Standard Deviation:", std_rating)

print("\n95% Confidence Interval for Mean Rating:")
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

# Customer satisfaction level
if mean_rating >= 4:
    satisfaction = "Highly Satisfied"
elif mean_rating >= 3:
    satisfaction = "Moderately Satisfied"
elif mean_rating >= 2:
    satisfaction = "Less Satisfied"
else:
    satisfaction = "Dissatisfied"

print("\nCustomer Satisfaction Level:", satisfaction)