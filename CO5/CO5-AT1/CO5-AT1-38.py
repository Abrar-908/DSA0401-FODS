speed = int(input("Enter vehicle speed: "))
vehicles = int(input("Enter number of vehicles: "))

severity = speed + vehicles * 10

if severity < 50:
    print("Low Severity")
elif severity < 100:
    print("Medium Severity")
else:
    print("High Severity")