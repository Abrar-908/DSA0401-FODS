import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --------------------------------

# 1. Load Dataset

# --------------------------------

df = pd.read\_csv("retail\_sales\_case\_study\_7.csv")

print("\nRETAIL SALES DATA\n")
print(df.head())

print("\nDATA INFORMATION\n")
print(df.info())

# --------------------------------

# 2. Descriptive Statistics

# --------------------------------

print("\nDESCRIPTIVE STATISTICS\n")

print(df[["Daily\_Sales", "Customer\_Count"]].describe())

print("\nMean Sales:")
print(df["Daily\_Sales"].mean())

print("\nMedian Sales:")
print(df["Daily\_Sales"].median())

print("\nStandard Deviation of Sales:")
print(df["Daily\_Sales"].std())

print("\nMean Customer Count:")
print(df["Customer\_Count"].mean())

print("\nMedian Customer Count:")
print(df["Customer\_Count"].median())

print("\nStandard Deviation of Customer Count:")
print(df["Customer\_Count"].std())

# --------------------------------

# 3. Distribution Analysis

# --------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(df["Daily\_Sales"], bins=20, kde=True)
plt.title("Distribution of Daily Sales")
plt.xlabel("Daily Sales")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["Customer\_Count"], bins=20, kde=True)
plt.title("Distribution of Customer Count")
plt.xlabel("Customer Count")
plt.ylabel("Frequency")
plt.show()

# --------------------------------

# 4. Boxplot for Sales

# --------------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Daily\_Sales"])
plt.title("Boxplot of Daily Sales")
plt.xlabel("Daily Sales")
plt.show()

# --------------------------------

# 5. IQR Outlier Detection

# --------------------------------

Q1 = df["Daily\_Sales"].quantile(0.25)
Q3 = df["Daily\_Sales"].quantile(0.75)

IQR = Q3 - Q1

lower\_limit = Q1 - 1.5 \* IQR
upper\_limit = Q3 + 1.5 \* IQR

print("\nIQR OUTLIER ANALYSIS")

print("Q1 =", Q1)
print("Q3 =", Q3)
print("IQR =", IQR)
print("Lower Limit =", lower\_limit)
print("Upper Limit =", upper\_limit)

outliers = df[
(df["Daily\_Sales"] < lower\_limit) |
(df["Daily\_Sales"] > upper\_limit)
]

print("\nOUTLIER RECORDS\n")
print(outliers)

print("\nNumber of Outliers:", len(outliers))

# --------------------------------

# 6. Z-Score Outlier Detection

# --------------------------------

df["Z\_Score"] = stats.zscore(df["Daily\_Sales"])

z\_outliers = df[abs(df["Z\_Score"]) > 3]

print("\nZ-SCORE OUTLIERS\n")
print(z\_outliers)

print("\nNumber of Z-Score Outliers:",
len(z\_outliers))

# --------------------------------

# 7. Urban and Rural Statistics

# --------------------------------

urban = df[df["Branch\_Type"] == "Urban"]["Daily\_Sales"]
rural = df[df["Branch\_Type"] == "Rural"]["Daily\_Sales"]

print("\nURBAN BRANCH STATISTICS")
print(urban.describe())

print("\nRURAL BRANCH STATISTICS")
print(rural.describe())

# --------------------------------

# 8. Confidence Interval

# --------------------------------

urban\_mean = urban.mean()
urban\_std = urban.std()
urban\_n = len(urban)

rural\_mean = rural.mean()
rural\_std = rural.std()
rural\_n = len(rural)

urban\_se = urban\_std / np.sqrt(urban\_n)
rural\_se = rural\_std / np.sqrt(rural\_n)

urban\_t = stats.t.ppf(0.975, urban\_n - 1)
rural\_t = stats.t.ppf(0.975, rural\_n - 1)

urban\_ci = (
urban\_mean - urban\_t \* urban\_se,
urban\_mean + urban\_t \* urban\_se
)

rural\_ci = (
rural\_mean - rural\_t \* rural\_se,
rural\_mean + rural\_t \* rural\_se
)

print("\n95% CONFIDENCE INTERVAL")

print("Urban Mean =", urban\_mean)
print("Urban 95% CI =", urban\_ci)

print("Rural Mean =", rural\_mean)
print("Rural 95% CI =", rural\_ci)

# --------------------------------

# 9. Hypothesis Testing

# --------------------------------

t\_stat, p\_two = stats.ttest\_ind(
urban,
rural,
equal\_var=False
)

# One-tailed test: Urban > Rural

p\_one = p\_two / 2 if t\_stat > 0 else 1 - p\_two / 2

print("\nHYPOTHESIS TESTING")

print("Urban Mean =", urban\_mean)
print("Rural Mean =", rural\_mean)

print("t-statistic =", t\_stat)
print("One-tailed p-value =", p\_one)

alpha = 0.05

if p\_one < alpha:
print("\nDecision: Reject H0")
print("Urban branches have significantly higher average sales.")
else:
print("\nDecision: Fail to reject H0")
print("There is insufficient evidence that urban branches have higher average sales.")

# --------------------------------

# 10. Branch Comparison Plot

# --------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
data=df,
x="Branch\_Type",
y="Daily\_Sales"
)

plt.title("Urban vs Rural Daily Sales")
plt.xlabel("Branch Type")
plt.ylabel("Daily Sales")
plt.show()

# --------------------------------

# 11. Average Sales by Branch Type

# --------------------------------

average\_sales = df.groupby(
"Branch\_Type"
)["Daily\_Sales"].mean()

print("\nAVERAGE SALES BY BRANCH TYPE\n")
print(average\_sales)

average\_sales.plot(
kind="bar",
figsize=(8, 5)
)

plt.title("Average Sales: Urban vs Rural")
plt.xlabel("Branch Type")
plt.ylabel("Average Daily Sales")
plt.xticks(rotation=0)
plt.show()

# --------------------------------

# 12. Customer Count vs Sales

# --------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
data=df,
x="Customer\_Count",
y="Daily\_Sales",
hue="Branch\_Type"
)

plt.title("Customer Count vs Daily Sales")
plt.xlabel("Customer Count")
plt.ylabel("Daily Sales")
plt.show()

# --------------------------------

# 13. Correlation Analysis

# --------------------------------

correlation = df[
[
"Daily\_Sales",
"Customer\_Count",
"Discount\_Percentage",
"Feedback\_Score"
]
].corr()

print("\nCORRELATION MATRIX\n")
print(correlation)

plt.figure(figsize=(8, 6))

sns.heatmap(
correlation,
annot=True,
cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.show()



this is the code remove unncessary things

dont remove graphs etc
