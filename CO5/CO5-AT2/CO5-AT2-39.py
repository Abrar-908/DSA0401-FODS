import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Read CSV
data = pd.read_csv("pharmacy_transactions.csv")

# Group items by transaction
transactions = data.groupby("TransactionID")["Item"].apply(
    list
).tolist()

# Convert transactions to binary format
encoder = TransactionEncoder()

encoded_data = encoder.fit(transactions).transform(transactions)

basket = pd.DataFrame(
    encoded_data,
    columns=encoder.columns_
)

print("Transaction Matrix:")
print(basket)

# Apriori
frequent_itemsets = apriori(
    basket,
    min_support=0.05,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

# Select important columns
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

print("\nStrong Association Rules:")
print(rules)

# Top rules
top_rules = rules.head(10)

# Visualization
plt.bar(
    range(len(top_rules)),
    top_rules["lift"]
)

plt.xlabel("Association Rules")
plt.ylabel("Lift")
plt.title("Strongest Pharmacy Associations")
plt.show()
