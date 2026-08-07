# ============================================
# Exploratory Data Analysis (EDA)
# Crop Yield Dataset
# ============================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Farm_Crop_Yield_Dataset.csv")

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

# Mean
print("\nMean")
print(df[["Rainfall_mm","Temperature","Fertilizer_kg","Crop_Yield_kg"]].mean())

# Median
print("\nMedian")
print(df[["Rainfall_mm","Temperature","Fertilizer_kg","Crop_Yield_kg"]].median())

# Minimum
print("\nMinimum")
print(df[["Rainfall_mm","Temperature","Fertilizer_kg","Crop_Yield_kg"]].min())

# Maximum
print("\nMaximum")
print(df[["Rainfall_mm","Temperature","Fertilizer_kg","Crop_Yield_kg"]].max())

# Standard Deviation
print("\nStandard Deviation")
print(df[["Rainfall_mm","Temperature","Fertilizer_kg","Crop_Yield_kg"]].std())

# --------------------------------------------
# 6. Soil Type Distribution
# --------------------------------------------
print("\nSoil Type Count")
print(df["Soil_Type"].value_counts())

# --------------------------------------------
# 7. Average Crop Yield by Soil Type
# --------------------------------------------
print("\nAverage Crop Yield by Soil Type")
print(df.groupby("Soil_Type")["Crop_Yield_kg"].mean())

# --------------------------------------------
# 8. Correlation Matrix
# --------------------------------------------
print("\nCorrelation Matrix")
print(df[["Rainfall_mm",
          "Temperature",
          "Fertilizer_kg",
          "Crop_Yield_kg"]].corr())

# --------------------------------------------
# 9. Bar Chart
# --------------------------------------------
plt.figure(figsize=(7,5))

sns.barplot(x="Soil_Type",
            y="Crop_Yield_kg",
            data=df)

plt.title("Average Crop Yield by Soil Type")
plt.xlabel("Soil Type")
plt.ylabel("Crop Yield (kg)")
plt.show()

# --------------------------------------------
# 10. Histograms
# --------------------------------------------
df[["Rainfall_mm",
    "Temperature",
    "Fertilizer_kg",
    "Crop_Yield_kg"]].hist(figsize=(10,8))

plt.suptitle("Distribution of Numerical Features")
plt.show()

# --------------------------------------------
# 11. Scatter Plot
# --------------------------------------------

# Rainfall vs Crop Yield
plt.figure(figsize=(6,4))
sns.scatterplot(x="Rainfall_mm",
                y="Crop_Yield_kg",
                hue="Soil_Type",
                data=df,
                s=100)

plt.title("Rainfall vs Crop Yield")
plt.show()

# Temperature vs Crop Yield
plt.figure(figsize=(6,4))
sns.scatterplot(x="Temperature",
                y="Crop_Yield_kg",
                hue="Soil_Type",
                data=df,
                s=100)

plt.title("Temperature vs Crop Yield")
plt.show()

# Fertilizer vs Crop Yield
plt.figure(figsize=(6,4))
sns.scatterplot(x="Fertilizer_kg",
                y="Crop_Yield_kg",
                hue="Soil_Type",
                data=df,
                s=100)

plt.title("Fertilizer vs Crop Yield")
plt.show()

# --------------------------------------------
# 12. Boxplots
# --------------------------------------------
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
sns.boxplot(y=df["Rainfall_mm"])
plt.title("Rainfall")

plt.subplot(2,2,2)
sns.boxplot(y=df["Temperature"])
plt.title("Temperature")

plt.subplot(2,2,3)
sns.boxplot(y=df["Fertilizer_kg"])
plt.title("Fertilizer")

plt.subplot(2,2,4)
sns.boxplot(y=df["Crop_Yield_kg"])
plt.title("Crop Yield")

plt.tight_layout()
plt.show()

# --------------------------------------------
# 13. Pair Plot
# --------------------------------------------
sns.pairplot(df,
             vars=["Rainfall_mm",
                   "Temperature",
                   "Fertilizer_kg",
                   "Crop_Yield_kg"],
             hue="Soil_Type")

plt.show()

# --------------------------------------------
# 14. Correlation Heatmap
# --------------------------------------------
plt.figure(figsize=(7,5))

sns.heatmap(df[["Rainfall_mm",
                "Temperature",
                "Fertilizer_kg",
                "Crop_Yield_kg"]].corr(),
            annot=True,
            cmap="coolwarm",
            linewidths=0.5)

plt.title("Correlation Heatmap")
plt.show()

# --------------------------------------------
# 15. Top 5 Highest Yield Farms
# --------------------------------------------
print("\nTop 5 Highest Yield Farms")

top_yield = df.sort_values(by="Crop_Yield_kg",
                           ascending=False)

print(top_yield[["Farm_ID",
                 "Soil_Type",
                 "Crop_Yield_kg"]].head())

# --------------------------------------------
# 16. EDA Summary
# --------------------------------------------
print("\n===================================")
print("EDA SUMMARY")
print("===================================")

print("Total Farms :", len(df))

print("\nHighest Crop Yield")
print(df.loc[df["Crop_Yield_kg"].idxmax()][["Farm_ID",
                                            "Crop_Yield_kg"]])

print("\nLowest Crop Yield")
print(df.loc[df["Crop_Yield_kg"].idxmin()][["Farm_ID",
                                           "Crop_Yield_kg"]])

print("\nAverage Crop Yield by Soil Type")
print(df.groupby("Soil_Type")["Crop_Yield_kg"].mean())

print("\nAverage Rainfall by Soil Type")
print(df.groupby("Soil_Type")["Rainfall_mm"].mean())

print("\nAverage Fertilizer Usage by Soil Type")
print(df.groupby("Soil_Type")["Fertilizer_kg"].mean())

print("\nEDA Completed Successfully.")