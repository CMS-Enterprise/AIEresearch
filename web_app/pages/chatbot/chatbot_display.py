
import dash_bootstrap_components as dbc
from dash import html
import dash

header = html.Div(
    [
        html.H1(["Medicare Handbook Chatbot"], style={"color":"#00395E"}),
        html.Hr(),
        html.H2('Overview'),
        html.P(
            "A simple UI for ""Question and Answer"" chats with an LLM as that has access to documentation for Medicare Manuals. Upon a query submission, the system will search data stores (graph, vectors) for supporting data, then submit the query, chat history, and context to an LLM to respond."
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
            ]),
            html.Tr([
                html.Td("Provided Contextual Data"),
                html.Td("This is the documentation the user uploads as a resource for the LLM. This documentation may be retained for human evaluation of model responses for up to 30 days. Thereafter, the data is removed."),
            ]),  
            html.Tr([
                html.Td("'Medicare and You 2024' Handbook"),
                html.Td("The 'Medicare & You' handbook is mailed to all Medicare households each fall. It includes a summary of Medicare benefits, rights, and protections; lists of available health and drug plans; and answers to frequently asked questions about Medicare."),
            ]),  
        ], id="info_table"),
    ]
)

eval_trulense = html.Div(
    [
        html.Table([
            html.Tr([
                html.Th("Term"),
                html.Th("Definition"),
            ]),
            html.Tr([
                html.Td("Answer Relevance"),
                html.Td("This metric evaluates how relevant the LLM''s response is to the original user input. It ensures that the final response provided by the application is a helpful answer to the user's question."),
            ]),
            html.Tr([
                html.Td("Context Relevance"),
                html.Td("Context relevance is crucial in the retrieval step of a RAG (Retrieval-Augmented Generation) application. It measures the relevance of each chunk of context to the input query. High context relevance is important to ensure that the information used by the LLM to form an answer is pertinent to the query at hand, preventing irrelevant information from leading to hallucinations in the generated response."),
            ]),
            html.Tr([
                html.Td("Groundedness"),
                html.Td("Groundedness refers to the degree to which the LLM's response is factually consistent with the provided or retrieved context. After retrieval, as the context is formed into an answer by an LLM, the model may exaggerate or stray from the facts. Evaluating groundedness involves verifying that the claims made in the response can be supported by evidence within the retrieved context. This helps ascertain that the application''s responses are based on accurate information and are free from fabricated content, up to the limit of the knowledge base's accuracy."),
            ]),
            html.Tr([
                html.Td("Latency (Seconds)"),
                html.Td("Latency measures the time it takes for the LLM to generate a response, from the moment the query is submitted to when the response is received. This is a performance metric that can influence the user experience, as lower latency is generally preferred for responsiveness."),
            ]),
            html.Tr([
                html.Td("Total Cost (USD)"),
                html.Td("Total cost refers to the monetary cost associated with generating a response using the LLM. It is likely calculated based on factors like computational resources used, the number of tokens generated, and the model's operational costs. This can help users and developers understand the financial implications of using the LLM at scale."),
            ]),
            html.Tr([
                html.Td("Mean Absolute Error"),
                html.Td("This is a statistical measure used to evaluate the quality of feedback or models. It represents the average of the absolute differences between the expected scores and the actual scores provided by the LLM across all test cases. A lower mean absolute error indicates higher accuracy and reliability of the model in providing expected outcomes."),
            ]),
            html.Tr([
                html.Td("qs_relevance"),
                html.Td("Although not directly defined in the documentation, based on the context, qs_relevance seems to be a specific function or method that assesses the relevance of the LLM's response to a query, similar to 'relevance' but potentially tailored to a particular aspect or measured using a specific methodology. It could be part of a feedback loop to evaluate and track feedback quality across different models or prompting schemes."),
            ])

        ], id="info_table"),
    ]
)

accordion = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                [
                    html.P(
                        "Our approach involves the use of a single LLM to answer any simple query or request from a user. The responses generated are a reflection the models capability to 'reason' and use provided contextual documentation to form a response. Our intention is to test the deployment of a Generative AI Tool using an LLM model within our CMS owned AWS environment. Our intention is to test the usefulness of LLMs with more 'realistic' circumstances to what Medicare beneficiaries would use. We hypothesized that either beneficiaries or their agents would prefer to have the ability to ask natural questions by voice and hear responses in the same method."
                    ),
                    html.P(
                        "We utilize Retrieval Augmentation Generation (RAG) for this demonstration and measure it's quality using both autonomous LLM reviews (from other options than the one we're using) and through human evaluation to measure the quality and performance of the model for our use cases."
                    ),                    
                ], title="Approach"
            ),
            dbc.AccordionItem(
                [
                    html.P(
                        "We are using the following models in this example for the prompt and responses. You can find additional information for each below:"
                    ),
                    model_listing,
                    html.P(
                        "Although not currently, provided, we use a number of 'smaller' models to identify useful context from documentation that is appended to queries send to the llm. Check back soon for additional details."
                    ) 
                ], title="Model"
            ),
            dbc.AccordionItem(
                [
                    datasets_listing
                ], title="Data"
            ),
            dbc.AccordionItem(
                [
                    html.H4("TruLens Evaluations"),
                    eval_trulense,
                    html.Iframe(
                        src="http://127.0.0.1:8666/", 
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
                    html.H4("Human Evaluation"),
                    html.P(
                        "Measures of accuracy, relevance, and breadth. For now, this information is stored offline and will be included later autonomously."
                    ),
                    html.Img(
                        src="../assets/eval_scores.png",
                        style={
                            "display": "block", 
                            "margin-left": "auto", 
                            "margin-right": "auto", 
                            "width": "50%"
                        }, 
                    )
                ], title="Evaluation"
            ),
            dbc.AccordionItem(
                "The tool is configured in the CMS Knowledge Management Platform (KMP) AI Workspace. It is housed on AWS.", title="Environment"
            ),
            dbc.AccordionItem(
                [
                    html.P(
                        [
                            "This is a one-time release, fast prototype project and not open to active collaboration at this time.  While we appreciate comments, we may not be able to respond to suggestions via our ", 
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
            height=900, 
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
