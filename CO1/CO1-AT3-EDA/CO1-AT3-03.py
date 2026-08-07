# ============================================
# Exploratory Data Analysis (EDA)
# Supermarket Product Sales Dataset
# ============================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Product_Sales_Dataset.csv")

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
# 6. Category Distribution
# --------------------------------------------
print("\nProducts in Each Category")
print(df["Category"].value_counts())

# --------------------------------------------
# 7. Average Values by Category
# --------------------------------------------
print("\nAverage Price by Category")
print(df.groupby("Category")["Price"].mean())

print("\nAverage Quantity Sold by Category")
print(df.groupby("Category")["Quantity_Sold"].mean())

print("\nAverage Discount by Category")
print(df.groupby("Category")["Discount"].mean())

print("\nAverage Revenue by Category")
print(df.groupby("Category")["Revenue"].mean())

# --------------------------------------------
# 8. Correlation Matrix
# --------------------------------------------
print("\nCorrelation Matrix")
print(df[["Price","Quantity_Sold","Discount","Revenue"]].corr())

# --------------------------------------------
# 9. Histograms
# --------------------------------------------
df[["Price","Quantity_Sold","Discount","Revenue"]].hist(figsize=(10,8))
plt.suptitle("Distribution of Numerical Features")
plt.show()

# --------------------------------------------
# 10. Boxplots
# --------------------------------------------
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
sns.boxplot(y=df["Price"])
plt.title("Price")

plt.subplot(2,2,2)
sns.boxplot(y=df["Quantity_Sold"])
plt.title("Quantity Sold")

plt.subplot(2,2,3)
sns.boxplot(y=df["Discount"])
plt.title("Discount")

plt.subplot(2,2,4)
sns.boxplot(y=df["Revenue"])
plt.title("Revenue")

plt.tight_layout()
plt.show()

# --------------------------------------------
# 11. Category Count Plot
# --------------------------------------------
plt.figure(figsize=(7,5))
sns.countplot(x="Category", data=df)
plt.title("Number of Products in Each Category")
plt.xticks(rotation=20)
plt.show()

# --------------------------------------------
# 12. Revenue by Category
# --------------------------------------------
plt.figure(figsize=(7,5))
sns.barplot(x="Category", y="Revenue", data=df)
plt.title("Revenue by Category")
plt.xticks(rotation=20)
plt.show()

# --------------------------------------------
# 13. Price vs Revenue
# --------------------------------------------
plt.figure(figsize=(7,5))
sns.scatterplot(x="Price", y="Revenue", hue="Category", data=df, s=100)
plt.title("Price vs Revenue")
plt.show()

# --------------------------------------------
# 14. Quantity Sold vs Revenue
# --------------------------------------------
plt.figure(figsize=(7,5))
sns.scatterplot(x="Quantity_Sold", y="Revenue", hue="Category", data=df, s=100)
plt.title("Quantity Sold vs Revenue")
plt.show()

# --------------------------------------------
# 15. Discount vs Revenue
# --------------------------------------------
plt.figure(figsize=(7,5))
sns.scatterplot(x="Discount", y="Revenue", hue="Category", data=df, s=100)
plt.title("Discount vs Revenue")
plt.show()

# --------------------------------------------
# 16. Pair Plot
# --------------------------------------------
sns.pairplot(df, hue="Category")
plt.show()

# --------------------------------------------
# 17. Correlation Heatmap
# --------------------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df[["Price","Quantity_Sold","Discount","Revenue"]].corr(),
            annot=True,
            cmap="coolwarm",
            linewidths=0.5)

plt.title("Correlation Heatmap")
plt.show()

# --------------------------------------------
# 18. Top 5 High-Selling Products
# --------------------------------------------
print("\nTop 5 High-Selling Products")
top_sales = df.sort_values(by="Quantity_Sold", ascending=False)
print(top_sales[["Product_ID","Category","Quantity_Sold"]].head())

# --------------------------------------------
# 19. Top 5 Revenue-Generating Products
# --------------------------------------------
print("\nTop 5 Revenue Generating Products")
top_revenue = df.sort_values(by="Revenue", ascending=False)
print(top_revenue[["Product_ID","Category","Revenue"]].head())

# --------------------------------------------
# 20. EDA Summary
# --------------------------------------------
print("\n====================================")
print("EDA SUMMARY")
print("====================================")

print("Total Products :", len(df))

print("\nHighest Selling Product")
print(df.loc[df["Quantity_Sold"].idxmax()][["Product_ID","Quantity_Sold"]])

print("\nHighest Revenue Product")
print(df.loc[df["Revenue"].idxmax()][["Product_ID","Revenue"]])

print("\nAverage Revenue by Category")
print(df.groupby("Category")["Revenue"].mean())

print("\nAverage Quantity Sold by Category")
print(df.groupby("Category")["Quantity_Sold"].mean())

print("\nEDA Completed Successfully.")