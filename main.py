import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from joblib import dump, load
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Chemin vers le fichier des messages
file_path = 'SMSSpamCollection.txt'

# Lire le fichier et créer un dataframe
with open(file_path, 'r', encoding='utf-8') as file:
    # Effectuer le split une fois à la rencontre du premier espace
    data = [line.strip().split('\t', 1) for line in file]

# DataFrame
df = pd.DataFrame(data, columns=['label', 'message'])
print(df.head())
