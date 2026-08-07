# ============================================
# Exploratory Data Analysis (EDA)
# Loan Approval Dataset
# ============================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Loan_Approval_Dataset.csv")

# --------------------------------------------
# 1. Display Dataset
# --------------------------------------------
print("\nFirst Five Records")
print(df.head())

print("\nLast Five Records")
print(df.tail())

# --------------------------------------------
# 2. Dataset Information
# --------------------------------------------
print("\nDataset Information")
print(df.info())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns)

# --------------------------------------------
# 3. Missing Values
# --------------------------------------------
print("\nMissing Values")
print(df.isnull().sum())

# --------------------------------------------
# 4. Duplicate Records
# --------------------------------------------
print("\nDuplicate Records")
print(df.duplicated().sum())

# --------------------------------------------
# 5. Statistical Summary
# --------------------------------------------
print("\nStatistical Summary")
print(df.describe())

# --------------------------------------------
# 6. Loan Status Distribution
# --------------------------------------------
print("\nLoan Status Count")
print(df["Loan_Status"].value_counts())

print("\nLoan Status Percentage")
print(df["Loan_Status"].value_counts(normalize=True) * 100)

# --------------------------------------------
# 7. Employment Type Distribution
# --------------------------------------------
print("\nEmployment Type Count")
print(df["Employment_Type"].value_counts())

# --------------------------------------------
# 8. Average Values by Loan Status
# --------------------------------------------
print("\nAverage Values by Loan Status")
print(df.groupby("Loan_Status")[["Income","Credit_Score","Loan_Amount"]].mean())

# --------------------------------------------
# 9. Correlation Matrix
# --------------------------------------------
# Convert Loan Status to Numeric
df["Loan"] = df["Loan_Status"].map({"Rejected":0, "Approved":1})

print("\nCorrelation Matrix")
print(df[["Income","Credit_Score","Loan_Amount","Loan"]].corr())

# --------------------------------------------
# 10. Histograms
# --------------------------------------------
df[["Income","Credit_Score","Loan_Amount"]].hist(figsize=(10,8))
plt.suptitle("Distribution of Numerical Features")
plt.show()

# --------------------------------------------
# 11. Boxplots
# --------------------------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
sns.boxplot(y=df["Income"])
plt.title("Income")

plt.subplot(1,3,2)
sns.boxplot(y=df["Credit_Score"])
plt.title("Credit Score")

plt.subplot(1,3,3)
sns.boxplot(y=df["Loan_Amount"])
plt.title("Loan Amount")

plt.tight_layout()
plt.show()

# --------------------------------------------
# 12. Loan Status Count Plot
# --------------------------------------------
plt.figure(figsize=(5,4))
sns.countplot(x="Loan_Status", data=df)
plt.title("Loan Approval Distribution")
plt.show()

# --------------------------------------------
# 13. Employment Type Count Plot
# --------------------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Employment_Type", data=df)
plt.title("Employment Type Distribution")
plt.xticks(rotation=15)
plt.show()

# --------------------------------------------
# 14. Income vs Loan Status
# --------------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x="Loan_Status", y="Income", data=df)
plt.title("Income vs Loan Status")
plt.show()

# --------------------------------------------
# 15. Credit Score vs Loan Status
# --------------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x="Loan_Status", y="Credit_Score", data=df)
plt.title("Credit Score vs Loan Status")
plt.show()

# --------------------------------------------
# 16. Loan Amount vs Loan Status
# --------------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x="Loan_Status", y="Loan_Amount", data=df)
plt.title("Loan Amount vs Loan Status")
plt.show()

# --------------------------------------------
# 17. Income vs Credit Score
# --------------------------------------------
plt.figure(figsize=(6,5))
sns.scatterplot(x="Income",
                y="Credit_Score",
                hue="Loan_Status",
                style="Employment_Type",
                s=100,
                data=df)

plt.title("Income vs Credit Score")
plt.show()

# --------------------------------------------
# 18. Pair Plot
# --------------------------------------------
sns.pairplot(df,
             vars=["Income","Credit_Score","Loan_Amount"],
             hue="Loan_Status")
plt.show()

# --------------------------------------------
# 19. Correlation Heatmap
# --------------------------------------------
plt.figure(figsize=(7,5))

sns.heatmap(df[["Income",
                "Credit_Score",
                "Loan_Amount",
                "Loan"]].corr(),
            annot=True,
            cmap="coolwarm",
            linewidths=0.5)

plt.title("Correlation Heatmap")
plt.show()

# --------------------------------------------
# 20. Top Approved Customers
# --------------------------------------------
print("\nApproved Customers")
print(df[df["Loan_Status"]=="Approved"][["Customer_ID",
                                         "Income",
                                         "Credit_Score",
                                         "Loan_Amount"]])

# --------------------------------------------
# 21. Rejected Customers
# --------------------------------------------
print("\nRejected Customers")
print(df[df["Loan_Status"]=="Rejected"][["Customer_ID",
                                         "Income",
                                         "Credit_Score",
                                         "Loan_Amount"]])

# --------------------------------------------
# 22. EDA Summary
# --------------------------------------------
print("\n===================================")
print("EDA SUMMARY")
print("===================================")

print("Total Customers :", len(df))

print("\nApproved Loans :", (df["Loan_Status"]=="Approved").sum())
print("Rejected Loans :", (df["Loan_Status"]=="Rejected").sum())

print("\nAverage Income by Loan Status")
print(df.groupby("Loan_Status")["Income"].mean())

print("\nAverage Credit Score by Loan Status")
print(df.groupby("Loan_Status")["Credit_Score"].mean())

print("\nAverage Loan Amount by Loan Status")
print(df.groupby("Loan_Status")["Loan_Amount"].mean())

print("\nEmployment Type vs Loan Status")
print(pd.crosstab(df["Employment_Type"], df["Loan_Status"]))

print("\nEDA Completed Successfully.")