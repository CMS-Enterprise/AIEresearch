
import dash_bootstrap_components as dbc
from dash import html

header = html.Div(
    [
        html.H1(["Foundational LLM Q&A"], style={"color":"#00395E"}),
        html.Hr(),
        html.H2('Overview'),
        html.P(
            "A simple UI for prompting an open-source LLM with a query without context."
        )
    ]
)

model_listing = html.Div(
    [
        html.Table([
            html.Tr([
                html.Th("Model"),
                html.Th("HF Model Card"),
                html.Th("Details"),
            ]),
            html.Tr([
                html.Td("Mistral-7B-Instruct-v0.2"),
                html.Td([
                    html.A("mistralai/Mistral-7B-Instruct-v0.1", href='https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1', target="_blank")
                ]),
                html.Td("Published September 2023."),
            ]),
            html.Tr([
                html.Td("Llama 2"),
                html.Td([
                    html.A("meta-llama/Llama-2-7b-chat", href='https://huggingface.co/meta-llama/Llama-2-7b-chat', target="_blank")
                ]),
                html.Td("Published July 2023.  Also available with 13 and 70 billion parameters."),
            ])
        ], id="info_table"),
    ]
)

datasets_listing = html.Div(
    [
        html.P("While this use case does not utilize any datasets to inform the model, it does create data in the form of:"),  
        html.Table([
            html.Tr([
                html.Th("Output"),
                html.Th("Description"),
            ]),
            html.Tr([
                html.Td("Logs"),
                html.Td("A collection of data points used to evaluate the model's quality and implementation performance. This includes query(prompt), response, latency, datetime, and occasionally additional meta data. No data is collected about users. This data collection is automated and required to use the application, you cannot opt out."),
            ])
        ], id="info_table"),
    ]
)

accordion = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                "Our approach involves the use of a single LLM to answer any simple query or request from a user. The responses generated are a reflection of the ""information"" stored within the model from training and without any additional context provided to support the response. Our intention is to test the deployment of a Generative AI Tool using an LLM model within our CMS owned AWS environment. This is the simplest version.", title="Approach"
            ),
            dbc.AccordionItem(
                [
                    html.P(
                        "We are using Mistral or Llama2 in this example. You can find additional information for each below:"
                    ),
                    model_listing
                ], title="Model"
            ),
            dbc.AccordionItem(
                [
                    datasets_listing
                ], title="Data"
            ),
            dbc.AccordionItem(
                "We are not currently performing evaluations on this demo. This page will be updates once metrics have been defined.", title="Evaluation"
            ),
            dbc.AccordionItem(
                "The tool is configured in the CMS Knowledge Management Platform (KMP) AI Workspace. It is housed on AWS.", title="Environment"
            ),
            dbc.AccordionItem(
                [
                    html.P(
                        [
                            "Please submit pull requests, ask to collaborate, or star our code on our  ", 
                            html.A("GitHub Repository", href='https://github.com/cms-enterprise/AIEresearch', target="_blank")
                        ]
                    )
#                    html.P(
#                        [
#                            "Mistral-7B-Instruct-v0.2, governed by ", 
#                            html.A("Appache License 2.0", href='https://github.com/openstack/mistral/blob/master/LICENSE', target="_blank")
#                        ]
#                    )
                ], title="Code Repository"
            ),
        ],
        start_collapsed=True
    ),
)


def display_qa():
    return [
        header, 
        html.Iframe(
            src="http://127.0.0.1:7860/", 
            height=800, 
            width=950, 
            style={
                "display": "block", 
                "margin-left": "auto", 
                "margin-right": "auto", 
                "margin-top": "30px",
                "margin-bottom": "30px",
                "width": "75%"
            }
        ),
        accordion
    ]