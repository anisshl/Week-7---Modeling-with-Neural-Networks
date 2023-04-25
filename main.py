import pandas as pd

# Chemin vers le fichier des messages
file_path = 'SMSSpamCollection.txt'

# Lire le fichier et créer un dataframe
with open(file_path, 'r', encoding='utf-8') as file:
    # Effectuer le split une fois à la rencontre du premier espace
    data = [line.strip().split('\t', 1) for line in file]

df = pd.DataFrame(data, columns=['label', 'message'])

# print(df.head())
