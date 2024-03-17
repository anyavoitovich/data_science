# Базовые
import os
import time
import numpy as np

# EDA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Предварительная обработка данных
import re
import spacy
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split

# Шаг 1: Загрузка данных в PyCaret
import pandas as pd
from pycaret.classification import *

# Загрузка данных
from pycaret.datasets import get_data
data = get_data('amazon')

# Шаг 2: Предварительная обработка текста
# Пример предварительной обработки текста (можно доработать по вашему усмотрению)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление пунктуации
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Токенизация
    tokens = word_tokenize(text)
    # Удаление стоп-слов
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Лемматизация
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Применение предобработки к столбцу с отзывами
data['processed_reviews'] = data['reviewText'].apply(preprocess_text)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['processed_reviews'])

# Преобразуем полученную матрицу в DataFrame и объединим с исходными данными
X_df = pd.DataFrame(X.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
data_processed = pd.concat([data, X_df], axis=1)

# Шаг 3: Использование PyCaret для классификации
# Инициализация PyCaret и создание эксперимента
exp = setup(data=data_processed, target='Positive', session_id=123,fix_imbalance=True, fold_strategy='batch')

# Сравнение нескольких моделей
best_model = compare_models()

# Обучение модели
final_model = finalize_model(best_model)

# Визуализация результатов
plot_model(final_model, plot='confusion_matrix')


