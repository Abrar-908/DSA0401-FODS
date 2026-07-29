import pandas as pd

# Sample DataFrame
data = {
    "Review": [
        "Good product",
        "Very good quality",
        "Good service",
        "Excellent product",
        "Good quality"
    ]
}

df = pd.DataFrame(data)

text = " ".join(df["Review"]).lower()

words = text.split()

frequency = {}

for word in words:
    if word in frequency:
        frequency[word] += 1
    else:
        frequency[word] = 1

print("Word Frequency Distribution:\n")
for word, count in frequency.items():
    print(word, ":", count)