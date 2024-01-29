import dash
from dash import html

def display_about():
#    background_image = html.Img(
#            src=dash.get_asset_url("home_page_hero .png"),
#            style={
#                "border-radius": 50,
#                "height": 800,
#                "margin-left": 5,
#                "float": "left",
#            },
#        )

    return html.Div([
        html.H1("About AI Explorers Demo Playground"),
        html.P("The AI Explorers Demo Playground ...")
    ])

