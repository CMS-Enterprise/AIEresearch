
import dash_bootstrap_components as dbc
from dash import html

header = html.Div(
    [
        html.H1(["Document Upload & Query Tool"], style={"color":"#00395E"}),
        html.Hr(),
        html.H2('Overview'),
        html.P(
            "A simple UI for uploading contextual documentation that is made available to an LLM as supporting ‘knowledge’ graphs and vectors. The UI may then be used to submit a prompt. The system will evaluate the documentation uploaded and autonomously identify helpful context sources to provide alongside the prompt to an LLM."
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
                html.Td("tbd"),
                html.Td("tbd"),
            ]),
            html.Tr([
                html.Td("Llama 2"),
                html.Td("tbd"),
                html.Td("tbd"),
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
                html.Td("A collection of data points used to evaluate the model’s quality and implementation performance. This includes query(prompt), response, latency, datetime, and occasionally additional meta data. No data is collected about users. This data collection is automated and required to use the application, you cannot opt out."),
            ]),
            html.Tr([
                html.Td("Provided Contextual Data"),
                html.Td("This is the documentation the user uploads as a resource for the LLM. This documentation may be retained for human evaluation of model responses for up to 30 days. Thereafter, the data is removed."),
            ]),            
        ], id="info_table"),
    ]
)

accordion = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                [
                html.P(
                    "Our approach involves the use of a single LLM to answer any simple query or request from a user. The responses generated are a reflection the models capability to ''reason'' and use provided contextual documentation to form a response. Our intention is to test the deployment of a Generative AI Tool using an LLM model within our CMS owned AWS environment."
                ),
                html.P(
                    "We utilize Retrieval Augmentation Generation (RAG) for this demonstration and measure it''s quality using both autonomous LLM reviews (from other options than the one we''re using) and through human evaluation to measure the quality and performance of the model for our use cases."
                )
                ], title="Approach"
            ),
            dbc.AccordionItem(
                [
                    html.P(
                        "We are using the following models in this example for the prompt and responses. You can find additional information for each below:"
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
                [
                    html.P(
                        "An evaluation is the process of assessing the chatbot's performance and effectiveness in understanding and responding to user questions accurately and helpfully. This involves analyzing metrics such as response accuracy, user satisfaction, and the ability of the chatbot to handle various types of inquiries based on the data. The following are evaluation results by model."
                    ),
                    html.Iframe(
                        src="http://127.0.0.1:8089/", 
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
                ], title="Evaluation"
            ),
            dbc.AccordionItem(
                "The tool is configured in the CMS Knowledge Management Platform (KMP) environment. The environment refers to the specific setup or configuration within which the chatbot operates, encompassing the software, hardware, and network systems that support its functionality. GPU: GPU?", title="Environment"
            ),
            dbc.AccordionItem(
                "Find the code on our GitHub. <link to GitHub>", title="Code Repository"
            ),
        ],
        start_collapsed=True
    ),
)


def display_doc_upload():
    return [
        header, 
        html.Iframe(
            src="http://127.0.0.1:7861/", 
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