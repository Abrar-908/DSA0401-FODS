from collections import Counter

print("=" * 55)
print("      MEASURES OF CENTRAL TENDENCY & DISPERSION")
print("=" * 55)
print("1. Discrete Data")
print("2. Continuous (Grouped) Data")

choice = int(input("\nEnter your choice (1 or 2): "))

# -------------------- DISCRETE DATA --------------------

if choice == 1:

    n = int(input("\nEnter the number of observations: "))

    data = []

    print("Enter the observations:")

    for i in range(n):
        data.append(float(input()))

    # Mean
    mean = sum(data) / n

    # Median
    data.sort()

    if n % 2 == 0:
        median = (data[n//2 - 1] + data[n//2]) / 2
    else:
        median = data[n//2]

    # Mode
    frequency = Counter(data)
    max_freq = max(frequency.values())

    modes = []

    for key, value in frequency.items():
        if value == max_freq:
            modes.append(key)

    if len(modes) == 1:
        mode = modes[0]
    else:
        mode = "No Unique Mode"

    # Variance
    variance = 0

    for x in data:
        variance += (x - mean) ** 2

    variance = variance / n

    # Standard Deviation
    std_dev = variance ** 0.5

    print("\n" + "=" * 55)
    print("RESULT")
    print("=" * 55)

    print("Mean                 =", round(mean, 2))
    print("Median               =", round(median, 2))
    print("Mode                 =", mode)

    print("Variance             =", round(variance, 2))
    print("Standard Deviation   =", round(std_dev, 2))

# -------------------- CONTINUOUS DATA --------------------

elif choice == 2:

    n = int(input("\nEnter the number of class intervals: "))

    lower = []
    upper = []
    freq = []

    print("\nEnter the class intervals and frequencies")

    for i in range(n):

        l = float(input("\nLower Limit : "))
        u = float(input("Upper Limit : "))
        f = int(input("Frequency   : "))

        lower.append(l)
        upper.append(u)
        freq.append(f)

    # Mid Values
    mid = []

    for i in range(n):
        mid.append((lower[i] + upper[i]) / 2)

    # Mean

    fx = []

    for i in range(n):
        fx.append(mid[i] * freq[i])

    total_frequency = sum(freq)

    mean = sum(fx) / total_frequency

    # Median

    cumulative_frequency = []

    total = 0

    for f in freq:
        total += f
        cumulative_frequency.append(total)

    N = total_frequency

    median_class = 0

    for i in range(n):
        if cumulative_frequency[i] >= N / 2:
            median_class = i
            break

    L = lower[median_class]

    h = upper[median_class] - lower[median_class]

    f = freq[median_class]

    if median_class == 0:
        CF = 0
    else:
        CF = cumulative_frequency[median_class - 1]

    median = L + (((N / 2) - CF) / f) * h

    # Mode

    modal_class = freq.index(max(freq))

    L = lower[modal_class]

    h = upper[modal_class] - lower[modal_class]

    f1 = freq[modal_class]

    if modal_class == 0:
        f0 = 0
    else:
        f0 = freq[modal_class - 1]

    if modal_class == n - 1:
        f2 = 0
    else:
        f2 = freq[modal_class + 1]

    denominator = (2 * f1 - f0 - f2)

    if denominator != 0:
        mode = L + ((f1 - f0) / denominator) * h
    else:
        mode = "Cannot Calculate"

    # Variance

    variance = 0

    for i in range(n):
        variance += freq[i] * ((mid[i] - mean) ** 2)

    variance = variance / total_frequency

    # Standard Deviation

    std_dev = variance ** 0.5

    print("\n" + "=" * 55)
    print("RESULT")
    print("=" * 55)

    print("Mean                 =", round(mean, 2))
    print("Median               =", round(median, 2))

    if isinstance(mode, float):
        print("Mode                 =", round(mode, 2))
    else:
        print("Mode                 =", mode)

    print("Variance             =", round(variance, 2))
    print("Standard Deviation   =", round(std_dev, 2))

# -------------------- INVALID CHOICE --------------------

else:
    print("\nInvalid Choice!")