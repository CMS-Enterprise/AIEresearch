
import dash_bootstrap_components as dbc
from dash import html

header = html.Div(
    [
        html.H1(["Medicare Handbook Chatbot"], style={"color":"#00395E"}),
        html.Hr(),
        html.H2('Overview'),
        html.P(
            "The Medicare Handbook Chatbot sources information from the ""Medicare and You 2024"" handbook and supports users in answering their Medicare questions. The Chatbot aims to be friendly, knowledgeable, and provides examples of questions that the user can ask. By interacting with the Chatbot, users can reduce the time they would spend reading the 128 page Handbook."
        )
    ]
)

datasets_listing = html.Div(
    [
        html.H4("Datasets"),
        html.P("A dataset refers to the structured collection of information extracted from a source, including text, tables, and figures, that the chatbot will use to understand and respond to user questions. The following are datasets we used in our tool."),  
        html.Table([
            html.Tr([
                html.Th("Dataset"),
                html.Th("Description"),
                html.Th("Data Use Agreement"),
                html.Th("Link to Source"),
            ]),
            html.Tr([
                html.Td("TBD"),
                html.Td("TBD"),
                html.Td("TBD"),
                html.Td("TBD"),
            ])
        ], id="info_table"),
    ]
)

datastores_listing = html.Div(
    [
        html.H4("Data Stores"),
        html.P("A data store is the repository or database system where a dataset is securely stored, managed, and accessed by the chatbot to retrieve information needed to answer user questions effectively. The following are data stores we used in our tool."),  
        html.Table([
            html.Tr([
                html.Th("Data Store"),
                html.Th("Description"),
                html.Th("Type"),
                html.Th("Link to Location"),
            ]),
            html.Tr([
                html.Td("TBD"),
                html.Td("TBD"),
                html.Td("TBD"),
                html.Td("TBD"),
            ])
        ], id="info_table"),
    ]
)

accordion = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                "Our approach involves the integration of __________ (specific NLP framework or tool) for understanding user queries, combined with __________ (ML framework or tool) for generating responses that accurately match the information found in the 'Medicare and You' 2024 handbook. The chatbot is hosted on __________ (cloud service provider), taking advantage of its __________ (specific cloud services) for optimal performance and reliability.", title="Approach"
            ),
            dbc.AccordionItem(
                "A model is a computational framework that processes and interprets human language to generate responses. It is built using machine learning algorithms that analyze vast amounts of text data to understand and mimic human conversational patterns. The following are models we tested in our tool.", title="Model"
            ),
            dbc.AccordionItem(
                [
                    datasets_listing, 
                    html.Br(),
                    datastores_listing,
                ], title="Data"
            ),
            dbc.AccordionItem(
                [
                    html.P
                    (
                        "An evaluation is the process of assessing the chatbot's performance and effectiveness in understanding and responding to user questions accurately and helpfully. This involves analyzing metrics such as response accuracy, user satisfaction, and the ability of the chatbot to handle various types of inquiries based on the data. The following are evaluation results by model."
                    ),
                    html.Iframe(
                        src="http://127.0.0.1:8080/", 
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
                [
                    html.P(
                        [
                            "Please submit pull requests, ask to collaborate, or star our code on our  ", 
                            html.A("GitHub Repository", href='https://github.com/cms-enterprise/AIEresearch', target="_blank")
                        ]
                    )
                ], title="Code Repository"
            ),
        ],
        start_collapsed=True
    ),
)


def display_chatbot():
    return [
        header, 
        html.Iframe(
            src="http://127.0.0.1:7863/", 
            height=400, 
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