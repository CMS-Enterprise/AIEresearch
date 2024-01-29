"""
This app creates a simple sidebar layout using inline style arguments and the
dbc.Nav component.

dcc.Location is used to track the current location, and a callback uses the
current location to render the appropriate page content. The active prop of
each NavLink is set automatically according to the current pathname. To use
this feature you must install dash-bootstrap-components >= 0.11.0.

For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plot.ly/urls
"""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, dash_table, Dash
import dash_bootstrap_components as dbc

# import pages
from pages.home import display_home
from pages.page_not_found import page_not_found
from pages.human_centered_ai import display_human_centered_ai
from pages.about import display_about
from pages.doc_upload.doc_upload_display import display_doc_upload
from pages.tts.tts_display import display_tts
from pages.qa.qa_display import display_qa
from pages.chatbot.chatbot_display import display_chatbot

# external JavaScript files
external_scripts = [
    {
        'src': 'https://gradio.s3-us-west-2.amazonaws.com/4.15.0/gradio.js',
        'type': 'module'
    }
]

app = dash.Dash(
    external_scripts=external_scripts,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#F2F2F2",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

sidebar = html.Div(
    [
        html.P("AI Explorers"),
        html.H2([
            "Demo Playground"
        ], style={
            "color":"#0071BC"
        }),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.H4([
            "App Demos"
        ], style={
            "color":"#0071BC"
        }),
        dbc.Nav(
            [
                dbc.NavLink("Foundational LLM Q&A", href="/qa", active="exact"),
                dbc.NavLink("Doc Upload & Query Tool", href="/docu", active="exact"),
                dbc.NavLink("Text-to-Speech", href="/tts", active="exact"),
                dbc.NavLink("Medicare Handbook Chatbot", href="/chatbot", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.H4([
            "Documentation"
        ], style={
            "color":"#0071BC"
        }),
        dbc.Nav(
            [
                dbc.NavLink("About", href="/about", active="exact"),
                dbc.NavLink("Human Centered AI Systems", href="/human-centered-ai", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.H4([
            "Contact Us"
        ], style={
            "color":"#0071BC"
        }),
        dbc.Nav(
            [
                #dbc.NavLink("Send Us Feedback", href="/feedback", active="exact"),
                dbc.NavLink("Find Us on Slack", href="/slack", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return display_home()

    elif pathname == "/qa":
        return display_qa()

    elif pathname == "/docu":
        return display_doc_upload()
    
    elif pathname == "/tts":
        return display_tts()

    elif pathname == "/chatbot":
        return display_chatbot()

    elif pathname == "/about":
        return display_about()

    elif pathname == "/human-centered-ai":
        return display_human_centered_ai()

    #elif pathname == "/feedback":
    #    return html.P("This is the content of the ""Send Us Feedback"" page!")

    elif pathname == "/slack":
        return html.P("This is the content of the ""Find Us on Slack"" page!")

    return page_not_found()

if __name__ == "__main__":
    app.run_server(port=8888)