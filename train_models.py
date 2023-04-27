import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import preprocess_text
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pickle

# Charger et prétraiter les données
data = pd.read_csv('SMSSpamCollection.txt', sep='\t', header=None, names=['label', 'message'])
data['message'] = data['message'].apply(preprocess_text)

# Créer un ensemble de données d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Sauvegarder X_test dans un fichier CSV pour plus tard (Test app)
X_test.to_csv('X_test.csv', index=False, header=True)

# Vectoriser les messages
#-----------------------------------------------
# TfidfVectorizer est une classe de la bibliothèque Python scikit-learn, utilisée pour convertir un ensemble de documents texte en 
# une matrice de caractéristiques numériques. Il le fait en calculant le score TF-IDF (Term Frequency-Inverse Document Frequency) 
# pour chaque terme dans chaque document.

# Le score TF-IDF est un produit de deux composants :

# 1) Term Frequency (TF) : C'est la fréquence d'un mot dans un document donné. 
# Plus un mot apparaît fréquemment dans un document, plus son score de fréquence de terme sera élevé.

# 2) Inverse Document Frequency (IDF) : C'est une mesure de l'importance d'un mot dans l'ensemble des documents. 
# Un mot qui apparaît dans de nombreux documents peut être considéré comme moins important car il est moins discriminant. 
# L'IDF est calculé en prenant le logarithme du rapport entre le nombre total de documents et le nombre de documents contenant le terme.

# Le score TF-IDF d'un terme dans un document est simplement le produit de sa fréquence de terme (TF) et de sa fréquence inverse de document (IDF).
# Doc : https://datascientest.com/tf-idf-intelligence-artificielle
#-----------------------------------------------
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Enregistrer le vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)


#Création des modèles les plus pourri de France pour tester !! :')
# Scikit-learn Model
model_sklearn = LogisticRegression()
model_sklearn.fit(X_train_vec, y_train)

# Enregistrer le modèle Scikit-learn
with open('model_sklearn.pkl', 'wb') as f:
    pickle.dump(model_sklearn, f)

# TensorFlow Model
model_tensorflow = Sequential()
model_tensorflow.add(Dense(units=128, activation='relu', input_dim=X_train_vec.shape[1]))
model_tensorflow.add(Dense(units=64, activation='relu'))
model_tensorflow.add(Dense(units=1, activation='sigmoid'))

model_tensorflow.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

y_train_binary = (y_train == 'spam').astype(int)
y_test_binary = (y_test == 'spam').astype(int)

model_tensorflow.fit(X_train_vec.toarray(), y_train_binary, epochs=10, batch_size=32, validation_split=0.1)
model_tensorflow.save('model_tensorflow')

# Évaluer les modèles
y_pred_sklearn = model_sklearn.predict(X_test_vec)
y_pred_tensorflow = (model_tensorflow.predict(X_test_vec.toarray()) > 0.5).astype(int).flatten()

print("Scikit-learn Model Accuracy:", accuracy_score(y_test, y_pred_sklearn))
print("TensorFlow Model Accuracy:", accuracy_score(y_test_binary, y_pred_tensorflow))
