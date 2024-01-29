import dash
from dash import html
import dash_bootstrap_components as dbc

def display_human_centered_ai():

    header = html.Div(
        [
            html.H1(["Human-Centered AI Systems"], style={"color":"#00395E"}),
            html.Hr(),
            html.H2(["What is Human-Centered AI?"], style={"color":"#00395E"}),
            html.P(
                "As CMS application development organizations (ADOs) pursue designing and developing AI-based software, it is important for teams to ensure that their AI systems are human-centered. Our team achieves this by using trustworthy and responsible AI practices, referencing accessibility guidelines, referencing industry standard practices for human-centered AI interactions, and through team collaboration."
            ),
            html.Img(
                src=dash.get_asset_url("human_centered_ui.png"),
                style={
                    #"border-radius": 50,
                    #"height": 400,
                    #"margin-left": 5,
                    #"float": "right",
                    "display": "block", 
                    "margin-left": "auto", 
                    "margin-right": "auto", 
                    "width": "50%"
                },
            ),
        ]
    )

    accordion = html.Div(
        dbc.Accordion(
            [
                dbc.AccordionItem([
                    html.P(
                        "Our trustworthy and responsible AI practices are rooted in government and industry standard guidelines. The Executive Order 13960 ""Promoting the Use of Trustworthy Artificial Intelligence in the Federal Government"" mandates that federal agencies consider principles of fairness, non-discrimination, openness, transparency, safety, and security when deploying AI technologies. Our team referenced the Health and Human Services Trustworthy AI Playbook's further explanations of these principles as well as the latest research that covers the evaluation of large language models (LLMs). Other tools and research that our team used to understand the trustworthiness of our models are Facebook AI Research (FAIR) and Trulens."
                    ),
                    html.Img(
                        src=dash.get_asset_url("HHS_trustworty_ai_principles.png"),
                        style={
                            "display": "block", 
                            "margin-left": "auto", 
                            "margin-right": "auto", 
                            "width": "50%"
                        },
                    ),
                    ], title="Trustworthy and Responsible AI"
                ),
                dbc.AccordionItem([
                    html.P(
                        "Trustworthy and responsible AI refers not just to the underlying models and AI systems, but also the way that humans would interact with the system. Our team used the Microsoft Human-AI Experience (HAX) Toolkit guidelines to determine how the Chatbot's features should work to ensure a human-centered AI experience."
                    ),
                    html.P(
                        "The following guidelines can be seen in the Chatbot's interface below:"
                    ),  
                    html.Table([
                        html.Tr([
                            html.Th("Guideline"),
                            html.Th("Description"),
                        ]),
                        html.Tr([
                            html.Td("G1. Make clear what the system can do."),
                            html.Td("Set expectations for what the system is designed for."),
                        ]),
                        html.Tr([
                            html.Td("G2. Make clear how well the system can do what it can do."),
                            html.Td("Let users know that the system can make mistakes."),
                        ]),
                        html.Tr([
                            html.Td("G3. Show contextually relevant information."),
                            html.Td("Let users see examples of what they can ask the Chatbot."),
                        ])
                    ], id="info_table"),
                    html.Img(
                        src=dash.get_asset_url("chatbot_app.png"),
                        style={
                            "display": "block", 
                            "margin-left": "auto", 
                            "margin-right": "auto", 
                            "width": "50%"
                        },
                    ),
                    ], title="Human-Centered AI Interactions"
                ),
                dbc.AccordionItem(
                    [
                        html.P(
                            "Human-centered AI encompasses accessibility. In the design and development of our AI apps as well as the Demo Playground website, we aim to meet Section 508 requirements by following the Web Content Accessibility Guidelines (WCAG) 2.1. Ensuring software is accessible occurs in the both the design and development phases of the software development lifecycle."
                        ),
                        html.P(
                            "The following are some tools that our team uses to ensure that our software is accessible."
                        ),  
                        html.Table([
                            html.Tr([
                                html.Th("Tool"),
                                html.Th("How to use it"),
                            ]),
                            html.Tr([
                                html.Td("A11y Project WCAG Checklist"),
                                html.Td("Use this with your team to plan for and review WCAG implementation."),
                            ]),
                            html.Tr([
                                html.Td("Access Guide"),
                                html.Td("Use this with your team as a friendly introduction to accessibility."),
                            ]),
                            html.Tr([
                                html.Td("Eight Shapes Contrast Grid"),
                                html.Td("Paste in hex codes to check foreground and background color contrast."),
                            ]),
                            html.Tr([
                                html.Td("Color Blindness Simulator"),
                                html.Td("Upload an image and view it as someone who has colorblindness."),
                            ]),
                            html.Tr([
                                html.Td("ANDI Accessibility Testing Tool"),
                                html.Td("Browser extension to test for accessibility developed by the Social Security Administration."),
                            ]),                            
                        ], id="info_table"),
                    ], title="Accessibility"
                ),
                dbc.AccordionItem(
                    [
                        html.P(
                            "Ensuring team alignment early in the process and scheduling team collaboration sessions are critical for the success of an accessible and human-centered AI software. Level setting with your team about the importance of accessibility by exploring Stories of Web Users may allow your team to more deeply understand different user personas and the effects of accessibility barriers. For human-centered AI, the Microsoft HAX Toolkit is comprehensive and provides directions on how a team can use it collaboratively to understand how AI guidelines can be applied to their software."
                        ),
                        html.Table([
                            html.Tr([
                                html.Th("Collaboration Step"),
                                html.Th("How to use it"),
                            ]),
                            html.Tr([
                                html.Td("Ensure alignment early & identify gaps"),
                                html.Td(
                                    [
                                        html.Ul([
                                            html.Li("Confirm that every team member understands the expectations of designing and developing an accessible and human-centered AI software."),
                                            html.Li("Use industry standard information and tools linked on this page to help team members understand what accessibility and human-centered AI entails."),
                                            html.Li("It is common for team members to have varying levels of understanding in regards to accessibility and human-centered AI. Referencing industry standards, setting expectations for the vision of the software, and having conversations early in the process can help prevent future misalignment."),
                                        ])
                                    ]
                                ),
                            ]),
                            html.Tr([
                                html.Td("Agree on communication processes"),
                                html.Td(                                    
                                    [
                                        html.Ul([
                                            html.Li("Decide on meeting and collaboration cadences early."),
                                            html.Li("Follow Agile methodologies and adopt ceremonies such as stand-ups and working sessions."),
                                        ])
                                    ]),
                            ]),
                            html.Tr([
                                html.Td("Emphasize collaboration and review"),
                                html.Td(
                                    [
                                        html.Ul([
                                            html.Li("Research, design, and development team members should have regular touchpoints to understand user needs, accessibility and human-centered AI standards, as well as any constraints.")
                                        ])
                                    ]
                                ),
                            ]),                           
                        ], id="info_table"),                        
                    ], title="Team Collaboration"
                ),
            ],
            start_collapsed=True
        ),
    )



    return [
        header, 
        accordion
    ]
