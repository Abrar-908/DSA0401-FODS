file = open("sample_text.txt", "r")

text = file.read().lower()
file.close()

words = text.split()

frequency = {}

for word in words:
    if word in frequency:
        frequency[word] += 1
    else:
        frequency[word] = 1

print("Word Frequency Distribution:\n")

for word in frequency:
    print(word, ":", frequency[word])