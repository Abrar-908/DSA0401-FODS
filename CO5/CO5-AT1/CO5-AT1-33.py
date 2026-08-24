usage = [100, 120, 90, 200, 110]

average = sum(usage) / len(usage)

print("Average Usage:", average)

for i in range(len(usage)):
    if usage[i] > average:
        print("House", i + 1, ": High Usage - Alert")
    else:
        print("House", i + 1, ": Normal")