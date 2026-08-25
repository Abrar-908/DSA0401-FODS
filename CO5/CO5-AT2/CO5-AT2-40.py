import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, apriori
from mlxtend.frequent_patterns import association_rules

# Read CSV
data = pd.read_csv("travel_bookings.csv")

# Convert each booking into a transaction
transactions = data.groupby("BookingID").agg(
    lambda x: list(x)
).apply(
    lambda row: [
        item
        for sublist in row
        for item in sublist
    ],
    axis=1
).tolist()

# Remove duplicate items
transactions = [
    list(set(transaction))
    for transaction in transactions
]

# Transaction encoding
encoder = TransactionEncoder()

encoded = encoder.fit(
    transactions
).transform(transactions)

basket = pd.DataFrame(
    encoded,
    columns=encoder.columns_
)

# FP-Growth
frequent_itemsets = fpgrowth(
    basket,
    min_support=0.05,
    use_colnames=True
)

print("Frequent Travel Itemsets:")
print(frequent_itemsets)

# Association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

rules = rules[
    [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]
]

# Sort by lift
rules = rules.sort_values(
    by="lift",
    ascending=False
)

print("\nStrong Travel Recommendations:")
print(rules.head(10))

# Visualization
top_rules = rules.head(10)

plt.bar(
    range(len(top_rules)),
    top_rules["confidence"]
)

plt.xlabel("Travel Package Rules")
plt.ylabel("Confidence")
plt.title("Strongest Travel Package Recommendations")
plt.show()

# -----------------------------
# Compare FP-Growth with Apriori
# -----------------------------

apriori_itemsets = apriori(
    basket,
    min_support=0.05,
    use_colnames=True
)

print("\nNumber of itemsets using FP-Growth:",
      len(frequent_itemsets))

print("Number of itemsets using Apriori:",
      len(apriori_itemsets))