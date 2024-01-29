from dash import html

def page_not_found():
    return html.Div([
        html.H1('Under Contruction'),
        html.H2('Demo currently in development!')
    ])