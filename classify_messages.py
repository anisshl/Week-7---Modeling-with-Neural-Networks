import pickle
from preprocessing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
import numpy as np

def classify_message(message, model_version):
    # Charger le modèle
    if model_version == 'scikit-learn-MLPClassifier':
        with open('model_sklearn_mlp.pkl', 'rb') as f:
            model = pickle.load(f)
    # elif model_version == 'scikit-learn-Perceptron': => pas de methode predict_proba pour le perceptron
    #     with open('model_sklearn_perceptron.pkl', 'rb') as f:
    #         model = pickle.load(f)
    elif model_version == 'scikit-learn-reg-log':
        with open('model_sklearn_reg_log.pkl', 'rb') as f:
            model = pickle.load(f)
    elif model_version == 'tensorflow':
        model = load_model('tensorflow_best_model.h5')
    else:
        raise ValueError("Invalid model version")

    # Charger le vectorizer
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Prétraiter le message
    preprocessed_message = preprocess_text(message)

    # Vectoriser le message
    message_vec = vectorizer.transform([preprocessed_message])

    prediction_proba = None
    prediction = None

    # Classer le message et obtenir la probabilité
    if 'scikit-learn' in model_version:
        # Exemple returned format : [[0.23133105 0.76866895]]
        prediction_proba = model.predict_proba(message_vec)
        prediction = np.argmax(prediction_proba, axis=1)

        prediction_proba = prediction_proba[0][prediction][0]

    elif model_version == 'tensorflow':
        # Exemple returned format : [[0.91266895]]
        prediction_proba = model.predict(message_vec.toarray())
        prediction = (prediction_proba > 0.5).astype(int).flatten()

        # Ham
        if prediction == 0 :
            prediction_proba = 1-prediction_proba[0][0]
        # Spam
        elif prediction == 1 :
            prediction_proba = prediction_proba[0][0]
        else:
            prediction_proba = None

    # Retourner la prédiction et la probabilité
    if prediction == 1:
        return 'Spam', prediction_proba
    if prediction == 0:
        return 'Ham', prediction_proba
    else:
        return 'Inconnue', prediction_proba

