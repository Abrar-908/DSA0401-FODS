import pandas as pd
import matplotlib.pyplot as plt
import string
from collections import Counter

# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\Downloads\data.csv")

text = " ".join(df["feedback"].astype(str)).lower()

text = text.translate(str.maketrans("", "", string.punctuation))

# Stop words
stop_words = {
    "the", "and", "is", "a", "an", "to", "of", "in",
    "on", "for", "with", "this", "that", "it"
}

words = []
for word in text.split():
    if word not in stop_words:
        words.append(word)

frequency = Counter(words)

N = int(input("Enter the number of top frequent words: "))

top_words = frequency.most_common(N)

print("\nTop", N, "Frequent Words:")
for word, count in top_words:
    print(word, ":", count)

word_list = [item[0] for item in top_words]
count_list = [item[1] for item in top_words]

plt.bar(word_list, count_list)
plt.title("Top Frequent Words")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()