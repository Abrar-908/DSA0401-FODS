weight = float(input("Enter product weight: "))
size = float(input("Enter product size: "))

if weight < 10 or weight > 20 or size < 5 or size > 15:
    print("Defective")
else:
    print("Non-Defective")