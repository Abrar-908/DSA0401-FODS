# Item prices and quantities
prices = [50, 100, 30]
quantities = [2, 1, 5]

# Discount and tax rates (in percentage)
discount_rate = 10   # 10%
tax_rate = 5         # 5%

# Calculate subtotal
subtotal = 0
for i in range(len(prices)):
    subtotal += prices[i] * quantities[i]

# Calculate discount
discount = subtotal * (discount_rate / 100)

# Price after discount
discounted_total = subtotal - discount

# Calculate tax
tax = discounted_total * (tax_rate / 100)

# Final total
final_total = discounted_total + tax

# Display results
print("Subtotal =", subtotal)
print("Discount =", discount)
print("Tax =", tax)
print("Total Cost =", final_total)