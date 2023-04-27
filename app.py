import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from classify_messages import classify_message

# Ajouter le support pour Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Créer la mise en page de l'application Dash
app.layout = html.Div([
    # Titre principal
    # La classe display-4 est utilisée pour styliser le titre principal et le rendre plus grand. 
    # La classe text-center centre le texte 
    # py-4 ajoute un espacement vertical (padding) autour du texte.
    html.H1("SMS Spam Detector", className='display-4 text-center py-4'),
    # Groupe de formulaires pour le champ de saisie du message
    html.Div([
        # form-control: Cette classe est utilisée pour styliser les champs de saisie et les rendre plus attrayants et cohérents avec le style Bootstrap. 
        # Elle ajoute également des styles pour les états de mise au point (focus) et les interactions.
        dcc.Textarea(id='input_message', placeholder='Enter a message...', className='form-control', style={'height': '150px'}),
    ], className='form-group'),
    # Saut de ligne
    html.Br(),
    # Groupe de formulaires pour la sélection du modèle
    html.Div([
        # form-label: Cette classe est utilisée pour styliser les étiquettes (labels) des éléments de formulaire, les rendant plus lisibles et cohérents avec le style Bootstrap.
        html.Label('Select model version:', className='form-label'),
        html.Br(),
        dcc.RadioItems(
            id='model_version',
            options=[
                {'label': 'Scikit-Learn', 'value': 'scikit-learn'},
                {'label': 'TensorFlow', 'value': 'tensorflow'}
            ],
            inline=False,
            value='scikit-learn'
        ),
    ], className='form-group'),
    # Saut de ligne
    html.Br(),
    # Groupe de formulaires pour le bouton de classification
    html.Div([
        # La classe btn est utilisée pour styliser les boutons
        # => btn-primary ajoute une couleur de fond bleue pour indiquer l'action principale. 
        # La classe w-100 ajuste la largeur du bouton pour qu'il occupe 100% de la largeur du conteneur parent.
        html.Button('Classify Message', id='classify_button', n_clicks=0, className='btn btn-primary w-100'),
    ], className='form-group'),
    # Saut de ligne
    html.Br(),
    # Div pour afficher la sortie (résultat de la classification)
    # La classe text-center centre le texte et py-3 ajoute un espacement vertical (padding) autour du texte.
    html.Div(id='output', className='form-group text-center py-3', style={'background-color': '#f8f9fa', 'border-radius': '5px'})
# La classe container est utilisée pour centrer et aligner le contenu de la page. 
# Elle ajoute également une marge horizontale automatique pour centrer le contenu. 
# La classe mt-4 ajoute une marge supérieure pour espacer le contenu du bord supérieur de la page.
], className='container mt-4')

# Callback pour mettre à jour la sortie en fonction des entrées utilisateur
@app.callback(
    Output('output', 'children'),
    [Input('classify_button', 'n_clicks')],
    [State('input_message', 'value'),
     State('model_version', 'value')]
)
def update_output(n_clicks, input_message, model_version):
    """
    text-warning, text-success et text-danger: Ces classes sont utilisées pour colorer le texte en fonction du résultat de la classification. 
    
    La classe text-warning est utilisée pour afficher un message d'avertissement lorsque l'utilisateur n'a pas entré de texte. 
    La classe text-success est utilisée pour les messages classifiés comme "ham" (non-spam).
    La classe text-danger est utilisée pour les messages classifiés comme "spam".

    Theme Dark Colors : https://codescandy.com/dashui/docs/colors.html
    """
    if not input_message:
        return html.Span('Please enter a message to classify.', className='text-warning')
    if n_clicks is not None:
        prediction, probability = classify_message(input_message, model_version)
        if prediction == 'ham':
            color = 'success'
        else:
            color = 'danger'
        return html.Span(f'The message is classified as {prediction} with a probability of {round(probability*100,2)} %', className='text-' + color)
    else:
        return ''

if __name__ == '__main__':
    app.run_server(debug=True)
