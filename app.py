import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from classify_messages import classify_message

# Ajouter le support pour Bootstrap
external_stylesheets = [
    'https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1("SMS Spam Detector", className='display-4 text-center py-4'),
    html.Div([
        dcc.Textarea(id='input_message', placeholder='Enter a message...', className='form-control', style={'height': '150px'}),
    ], className='form-group'),
    html.Div([
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
    html.Div([
        html.Button('Classify Message', id='classify_button', n_clicks=0, className='btn btn-primary w-100'),
    ], className='form-group'),
    html.Div(id='output', className='form-group text-center py-3', style={'background-color': '#f8f9fa', 'border-radius': '5px'})
], className='container mt-4')

@app.callback(
    Output('output', 'children'),
    [Input('classify_button', 'n_clicks')],
    [State('input_message', 'value'),
     State('model_version', 'value')]
)
def update_output(n_clicks, input_message, model_version):
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
