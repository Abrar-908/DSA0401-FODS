passengers = [500, 800, 1200, 700, 600]

peak = max(passengers)
period = passengers.index(peak) + 1

print("Peak Passengers:", peak)
print("Peak Period:", period)
print("Future Passenger Prediction:", sum(passengers) // len(passengers))