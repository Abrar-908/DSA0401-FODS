# ============================================
# Find Top 5 Most Sold Products Using Pandas
# ============================================

import pandas as pd

# Sample Data
data = {
    "Product_Name": [
        "Laptop", "Mouse", "Keyboard", "Laptop",
        "Mouse", "Laptop", "Monitor", "Keyboard",
        "Mouse", "Laptop", "Printer", "Mouse"
    ],
    "Quantity_Sold": [2, 5, 3, 4, 6, 3, 2, 5, 4, 2, 1, 3]
}

# Create DataFrame
sales_data = pd.DataFrame(data)

# Display DataFrame
print("Sales Data")
print(sales_data)

# --------------------------------------------
# Find Top 5 Products Sold the Most
# --------------------------------------------
top_5_products = (
    sales_data.groupby("Product_Name")["Quantity_Sold"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

print("\nTop 5 Products Sold in the Past Month")
print(top_5_products)