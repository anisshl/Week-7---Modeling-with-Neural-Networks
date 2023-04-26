import spacy
import pandas as pd

# Charger le modèle de langue anglaise
nlp = spacy.load("en_core_web_sm")

# importing the module
import json

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
 
# Opening JSON file
with open('dictionnary_supp.json') as dictionnary_json_file:
    dict_supp_voc = json.load(dictionnary_json_file)

def expand_abbreviations(text):
    """
    Remplace les abréviations courantes par leurs formes complètes.

    :param text: str, texte contenant des abréviations
    :return: str, texte avec les abréviations étendues
    """

    words = text.split()
    expanded_words = []

    for word in words:
        normalized_word = word.lower()
        if normalized_word in list(dict_supp_voc["abbreviations"].keys()):
            expanded_words.append(dict_supp_voc["abbreviations"][normalized_word])
        else:
            expanded_words.append(word)

    return ' '.join(expanded_words)

def expand_slang(text):
    """
    Remplace les mots d'argot et leurs variantes par des mots normalisés.

    :param text: str, texte contenant des mots d'argot
    :return: str, texte avec les mots d'argot étendus
    """

    words = text.split()
    expanded_words = []

    for word in words:
        normalized_word = word.lower()
        if normalized_word in list(dict_supp_voc["slang"].keys()):
            expanded_words.append(dict_supp_voc["slang"][normalized_word])
        else:
            expanded_words.append(word)

    return ' '.join(expanded_words)

# Fonction pour prétraiter le texte
def preprocess_text(text):
    """
    Prend en entrée une chaîne de texte et effectue les opérations de prétraitement suivantes :
    - Tokenisation

    - Lemmatisation : Notez que la lemmatisation est généralement préférée au stemming, car elle produit des résultats plus précis en tenant compte du contexte linguistique.
    => Lemmatisation more infos : https://fr.wikipedia.org/wiki/Lemmatisation#:~:text=La%20lemmatisation%20d%C3%A9signe%20un%20traitement,index%20ou%20de%20son%20analyse.

    - Suppression des mots vides

    - Suppression de la ponctuation

    :param text: str, texte à prétraiter
    :return: str, texte prétraité
    """

    # Étendre les abréviations
    text = expand_abbreviations(text)

    # Étendre les mots d'argot
    text = expand_slang(text)

    # Crée un document spaCy à partir du texte
    doc = nlp(text)

    # Liste pour stocker les tokens prétraités
    preprocessed_tokens = []

    for token in doc:
        # Vérifie si le token est un mot vide ou un signe de ponctuation
        if not (token.is_stop or token.is_punct):
            # Lemmatise le token et l'ajoute à la liste
            preprocessed_tokens.append(token.lemma_.lower())

    # Convertit la liste de tokens en une chaîne de texte
    text = ' '.join(preprocessed_tokens)

    # Supprimer les caractères non alphabétiques
    text = re.sub(r'\W', ' ', text)

    # Supprimer les espaces supplémentaires
    text = re.sub(r'\s+', ' ', text).strip()

    preprocessed_text = text

    return preprocessed_text
