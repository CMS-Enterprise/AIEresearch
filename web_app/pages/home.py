import dash
from dash import html

def display_home():
    background_image = html.Img(
            src=dash.get_asset_url("home_page_hero.png"),
            style={
                "display": "block", 
                "margin-left": "auto", 
                "margin-right": "auto", 
                "width": "75%"
            },
        )

    return html.Div([
        background_image,
        html.Br(),
        html.H1("AI Explorers Demo Playground"),
        html.P("The AI Explorers Program is an effort by the Office of Information Technology that aims to create opportunities for federal employees at CMS to explore artificial intelligence (AI). The AI Explorers Demo Playground is a repository of AI prototypes designed and developed by our Research and Development team that you can explore and tinker with. By sharing these prototypes, we hope to create a collaborative space for learning and iterating on AI technologies.")
    ])

