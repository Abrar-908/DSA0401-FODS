# ============================================
# Pandas DataFrame Operations - Property Data
# ============================================

import pandas as pd

# --------------------------------------------
# Create Sample DataFrame
# --------------------------------------------
data = {
    "Property_ID": [101, 102, 103, 104, 105, 106],
    "Location": ["Chennai", "Bangalore", "Chennai", "Hyderabad", "Bangalore", "Chennai"],
    "Bedrooms": [3, 5, 4, 6, 2, 5],
    "Area_sqft": [1500, 2500, 1800, 3200, 1200, 2800],
    "Listing_Price": [7500000, 12000000, 9000000, 15000000, 6500000, 13500000]
}

# Create DataFrame
property_data = pd.DataFrame(data)

# Display DataFrame
print("Property Data")
print(property_data)

# --------------------------------------------
# 1. Average Listing Price in Each Location
# --------------------------------------------
avg_price = property_data.groupby("Location")["Listing_Price"].mean()

print("\nAverage Listing Price by Location")
print(avg_price)

# --------------------------------------------
# 2. Number of Properties with More Than 4 Bedrooms
# --------------------------------------------
count_properties = property_data[property_data["Bedrooms"] > 4].shape[0]

print("\nNumber of Properties with More Than 4 Bedrooms:", count_properties)

# --------------------------------------------
# 3. Property with the Largest Area
# --------------------------------------------
largest_property = property_data.loc[property_data["Area_sqft"].idxmax()]

print("\nProperty with the Largest Area")
print(largest_property)