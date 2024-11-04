import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Прочитати текст з файлу
with open('input_text.txt', 'r', encoding='utf-8') as f:
    content = f.read()# У тексті 94 слова

# Токенізація по словам
tokens = word_tokenize(content)

# Лемматизація та стеммінг
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
stemmed_tokens = [stemmer.stem(token) for token in tokens]

# Видалення стоп-слів
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]

# Видалення пункуації
filtered_tokens = [word for word in filtered_tokens if word not in string.punctuation]

# Записати оброблений текст у інший файл
with open('processed_text.txt', 'w', encoding='utf-8') as f:
    f.write(' '.join(filtered_tokens))

print("Оброблений текст збережено у файлі 'processed_text.txt'")
