import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import preprocess_text #, nlp_preprocess
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV

# from keras.wrappers.scikit_learn import KerasClassifier # (Ancienne version qui fonctionne mais dépréciée)
from scikeras.wrappers import KerasClassifier
import pickle

from tensorflow.keras.models import save_model

from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score

# Charger et prétraiter les données
data = pd.read_csv('SMSSpamCollection.txt', sep='\t', header=None, names=['label', 'message'])

# data = nlp_preprocess(data)

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



#################################################################################
# 1 )

# Scikit-learn LogisticRegression
model_sklearn = LogisticRegression()
model_sklearn.fit(X_train_vec, y_train)

# Enregistrer le modèle Scikit-learn
with open('model_sklearn_reg_log.pkl', 'wb') as f:
    pickle.dump(model_sklearn, f)

# Évaluer le modèle
y_pred_sklearn = model_sklearn.predict(X_test_vec)
#################################################################################

#################################################################################
# 2 )
# Scikit-learn Perceptron Model

ppn = Perceptron(random_state=0)
ppn.fit(X_train_vec, y_train)

# Enregistrer le modèle Scikit-learn Perceptron
with open('model_sklearn_perceptron.pkl', 'wb') as f:
    pickle.dump(ppn, f)

# Évaluer le modèle
y_pred_ppn = ppn.predict(X_test_vec)
#################################################################################

#################################################################################
# 3 )
#Scikit-learn Multi Layer Perceptron

mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_vec, y_train)

# Enregistrer le modèle Scikit-learn Multi Layer Perceptron
with open('model_sklearn_mlp.pkl', 'wb') as f:
    pickle.dump(mlp, f)

# Évaluer le modèle
y_pred_mlp = mlp.predict(X_test_vec)
#################################################################################

#################################################################################
# 4 )
# TensorFlow Model

def create_model(neurons=64, hidden_layers=1, activation='relu', optimizer='adam', learning_rate=0.001, momentum=0.0, input_dim=X_train_vec.shape[1]):
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_dim=input_dim))
    
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation=activation))
    
    model.add(Dense(units=1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.get(optimizer)
    optimizer.learning_rate.assign(learning_rate)
    optimizer

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

# Define the grid search parameters
param_grid = {
    'model__neurons': [32,64],
    'model__hidden_layers': [2, 3],
    'model__activation': ['relu','tanh'],
    'model__optimizer': ['adam','rmsprop'],
    'model__learning_rate': [0.01, 0.1],
    # 'model__momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
}

model_tensorflow = KerasClassifier(model=create_model, epochs=10, batch_size=32, verbose=2)

# Create a custom "scorer" object for the AUC-ROC
auc_roc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)

grid = GridSearchCV(estimator=model_tensorflow, param_grid=param_grid, scoring=auc_roc_scorer, n_jobs=1, cv=3, verbose=2)

y_train_binary = (y_train == 'spam').astype(int)
y_test_binary = (y_test == 'spam').astype(int)

# Train the model with the grid search
grid_result = grid.fit(X_train_vec.toarray(), y_train_binary)

# Retrieve the best model from the grid search result
best_model = grid_result.best_estimator_

print(type(best_model))

# Save the TensorFlow/Keras model
save_model(best_model.model_, 'tensorflow_best_model.h5')
# best_model.model.save('tensorflow_best_model.h5')

# Extraire les meilleurs paramètres
best_params = grid_result.best_params_

print("Best parameters:", best_params)

# Sauvegarder les meilleurs paramètres
with open('tensorflow_best_model_best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

# Evaluate the model
# y_pred_tensorflow = (best_model.model.predict(X_test_vec.toarray()) > 0.5).astype(int).flatten() # avec from keras.wrappers.scikit_learn import KerasClassifier
y_pred_tensorflow = (best_model.model_.predict(X_test_vec.toarray()) > 0.5).astype(int).flatten()

# Results
print("Scikit-learn Model Accuracy:", accuracy_score(y_test, y_pred_sklearn))
print("Scikit-learn Perceptron Model Accuracy:", accuracy_score(y_test, y_pred_ppn))
print("Scikit-learn MLPClassifier Model Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("TensorFlow Model Accuracy:", accuracy_score(y_test_binary, y_pred_tensorflow))
