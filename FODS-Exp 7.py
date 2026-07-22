# ============================================
# Pandas DataFrame Operations - E-commerce Orders
# ============================================

# Import Pandas
import pandas as pd

# --------------------------------------------
# Create Sample DataFrame
# --------------------------------------------
data = {
    "Customer_ID": ["C101", "C102", "C101", "C103", "C102", "C101", "C104"],
    "Order_Date": [
        "2024-01-05",
        "2024-01-08",
        "2024-01-15",
        "2024-02-01",
        "2024-02-10",
        "2024-03-05",
        "2024-03-15"
    ],
    "Product_Name": [
        "Laptop",
        "Mouse",
        "Laptop",
        "Keyboard",
        "Mouse",
        "Monitor",
        "Laptop"
    ],
    "Order_Quantity": [1, 2, 1, 3, 4, 2, 1]
}

# Create DataFrame
order_data = pd.DataFrame(data)

# Convert Order_Date to datetime format
order_data["Order_Date"] = pd.to_datetime(order_data["Order_Date"])

# Display DataFrame
print("Order Data")
print(order_data)

# --------------------------------------------
# 1. Total number of orders made by each customer
# --------------------------------------------
orders_per_customer = order_data.groupby("Customer_ID").size()

print("\nTotal Number of Orders by Each Customer")
print(orders_per_customer)

# --------------------------------------------
# 2. Average order quantity for each product
# --------------------------------------------
avg_quantity = order_data.groupby("Product_Name")["Order_Quantity"].mean()

print("\nAverage Order Quantity for Each Product")
print(avg_quantity)

# --------------------------------------------
# 3. Earliest and Latest Order Dates
# --------------------------------------------
earliest_date = order_data["Order_Date"].min()
latest_date = order_data["Order_Date"].max()

print("\nEarliest Order Date:", earliest_date.date())
print("Latest Order Date:", latest_date.date())