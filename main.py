import nltk
from nltk.corpus import gutenberg, stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import string
import matplotlib.pyplot as plt

nltk.download('gutenberg')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Завантажуємо текст bible-kjv.txt
text = gutenberg.raw('bible-kjv.txt')

# Токенізація та визначення кількості слів
tokens = word_tokenize(text)
num_words = len(tokens)
print(f"Загальна кількість слів: {num_words}")

# Визначення 10 найбільш вживаних слів
fdist = FreqDist(tokens)
most_common_words = fdist.most_common(10)

# Виведення 10 найбільш вживаних слів і їх кількості у консоль
print("10 найбільш вживаних слів:")
for word, frequency in most_common_words:
    print(f"{word}: {frequency}")

# Побудова стовпчастої діаграми
words, frequencies = zip(*most_common_words)
plt.figure(figsize=(10, 6))
plt.bar(words, frequencies, color='skyblue')
plt.title("10 найбільш вживаних слів")
plt.xlabel("Слова")
plt.ylabel("Частота")
plt.show()

# Видалення стоп-слів і пунктуації
stop_words = set(stopwords.words('english'))
filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word not in string.punctuation]

# Визначення 10 найбільш вживаних слів для очищеного тексту
fdist_filtered = FreqDist(filtered_tokens)
most_common_filtered_words = fdist_filtered.most_common(10)

# Виведення 10 найбільш вживаних слів після очищення
print("\n10 найбільш вживаних слів після очищення:")
for word, frequency in most_common_filtered_words:
    print(f"{word}: {frequency}")

# Побудова стовпчастої діаграми для очищеного тексту
filtered_words, filtered_frequencies = zip(*most_common_filtered_words)
plt.figure(figsize=(10, 6))
plt.bar(filtered_words, filtered_frequencies, color='lightgreen')
plt.title("10 найбільш вживаних слів після очищення")
plt.xlabel("Слова")
plt.ylabel("Частота")
plt.show()

