import dash
from dash import html

def display_about():
    return html.Div([
        html.H1("About AI Explorers Demo Playground"),
        html.P("The AI Explorers Demo Playground ...")
    ])

