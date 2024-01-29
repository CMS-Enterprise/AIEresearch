# Noblis ESI change for AIE
'''
Instuction tuning guide: https://www.philschmid.de/instruction-tune-llama-2

TODO:
* Print source data? Optionally?
* Have TruLens use a local model.
* Change embedding model to local. 
'''

# Import custom functions
import sys
sys.path.insert(1, '/mnt/efs/data/AIEresearch/')
import aie_helper_functions as aie_helper

# Imports
import os
import argparse
from threading import Thread
from typing import Iterator
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import torch
from transformers import TextIteratorStreamer

# RAG
from llama_index import (SimpleDirectoryReader, 
                         Document, 
                         ServiceContext, 
                         VectorStoreIndex)
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# TrueLens
import configparser
from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI
import numpy as np

# Test Imports
import json
import uuid
import pandas as pd
import csv


# Set global variables
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

LICENSE = """
<p/>

---
As a derivate work of [Llama-2-13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat) by Meta,
this demo is governed by the original [license](https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat/blob/main/LICENSE.txt) and [acceptable use policy](https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat/blob/main/USE_POLICY.md).
"""


# Create description
def create_description(size):
    '''
    Returns text description used in model page for model being used. 

    Args:
        size (int): 7 for 7B, 13 for 13B, and 70 for 70B.

    Returns:
        DESCRIPTION (str): Text used in model page. 
    '''
    DESCRIPTION = f"""\
    # Llama-2 {size}B Chat

    This Space demonstrates model [Llama-2-{size}b-chat](https://huggingface.co/meta-llama/Llama-2-{size}b-chat) by Meta, a Llama 2 model with {size}B parameters fine-tuned for chat instructions. 

    🔎 For more details about the Llama 2 family of models and how to use them with `transformers`, take a look [at this Hugging Face blog post](https://huggingface.co/blog/llama2).

    """
    return DESCRIPTION


def load_llama2_rag(model, tokenizer, lst_docs, streaming=True):
    '''
    Return query_engine for given documents. 

    Args:
        model (transformers.models.llama.modeling_llama.LlamaForCausalLM): Llama2 model (any size).
        tokenizer (transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast): Llama2 tokenizer.
        lst_doct (list): List of documents to be used in RAG model. 
        streaming (bool): True to set a streaming model, False otherwise. Defaults to True. 

    Returns:
        query_engine (type)
    
    TODO:
        Make embedding model and input. 
    '''
    # Read in PDF(s) into a llamaindex document object
    documents = SimpleDirectoryReader(input_files=[path_handbook_2024]).load_data()
    
    # Print document stats
    print()
    print(f"Documents type: {type(documents)}")
    print(f"Number of pages: {len(documents)}")
    print(f"Sub-document type: {type(documents[0])}")
    print(f"{documents[0]}\n")
    print()
    
    # Merge documents into one
    document = Document(text="\n\n".join([doc.text for doc in documents]))

    # Set the system prompt
    system_prompt = '''
    <s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as 
    helpfully as possible, while being safe. Your answers should not include
    any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
    Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain 
    why instead of answering something not correct. If you don't know the answer 
    to a question, please don't share false information.
    [/INST]
    '''
    # <</SYS>>
    # """

    # Create a HF LLM using the llama index wrapper 
    llm = HuggingFaceLLM(context_window=4096,
                         max_new_tokens=2048, #256,
                         system_prompt=system_prompt,
                         #  query_wrapper_prompt=query_wrapper_prompt,
                         model=model,
                         tokenizer=tokenizer, 
                         device_map='auto')

    # Create and dl embeddings instance  
    embeddings = LangchainEmbedding(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    print('embeddings loaded.\n')

    # Create new service context instance
    service_context = ServiceContext.from_defaults(chunk_size=1024,
                                                   llm=llm,
                                                   embed_model=embeddings)
                                                #    embed_model='local')

    # Create the index
    index = VectorStoreIndex.from_documents([document],
                                            service_context=service_context)
    print('index created.\n')

    # Create the query engine
    query_engine = index.as_query_engine(streaming=streaming)
    print('query engine loaded.\n')

    # Return 
    return query_engine

    # query function using rag model
def query_index(message, chat_history):
    # async def query_index(message, chat_history): # Approach to get async to work 
        # Add to log
    with open(log_file_path, 'a') as f:
        f.write(f"QUERY:\n{message}\n\n")
        f.write(f"RESPONSE:\n")

        # Set the streaming argument based on whether TruLens in active
    if args.trulens:
        # Set up TruLens context manager
        with tru_query_engine_recorder as recording:
            # Get non-streaming response
            response = query_engine.query(message) 
    else:
        # Get streaming response 
        response = query_engine.query(message)

    # Stream response to GUI        
    outputs = []
    # Set response object based on if trulens is working
    resp_obj = response.response if args.trulens else response.response_gen
    for text in resp_obj:
        outputs.append(text)
        yield "".join(outputs)
        # Add to log
        with open(log_file_path, 'a') as f:
            f.write(f"{text}")

    # Add to log
    with open(log_file_path, 'a') as f:
        # CONDISER CHANGING PARAMETERS HERE
        # /mnt/efs/data/aivenv/lib/python3.9/site-packages/llama_index/response/schema.py
        # https://docs.llamaindex.ai/en/stable/api_reference/response.html
        f.write(f"\n\nSOURCES:\n{response.get_formatted_sources()}\n\n")
        f.write('*'*100)
        f.write('\n\n')    

if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser(prog='app.py',
                                     description='Run Hugging Face Llama2 model on browser.')
    # Add argument for size
    parser.add_argument('-s', '--size', type=int, default=13, choices=[7, 13, 70],
                        help='integer in billions: [7, 13, 70]')
    # Add argument for name to use in logs
    parser.add_argument('-l', '--log_name', type=str, required=True,
                        help='String name to be used in log file.')
    # Add argument for using TruLens
    parser.add_argument('-t', '--trulens', required=False, action='store_true',
                        help='Add this flag to use TruLens.')
    # Parser args
    args = parser.parse_args()
 
    # Create desscription 
    description = create_description(args.size)

    # Set up log file
    log_file_path_save = '/mnt/efs/data/AIEresearch/demo_medicare_handbook/chat_logs'
    log_file_path = aie_helper.set_up_log_file(args.log_name, log_file_path_save)
    
    # Create model and tokenizer objects
    model, tokenizer = aie_helper.load_llama2_model(args.size)
    # Adjust tokenizer for use here
    tokenizer.use_default_system_prompt = False

    # Set up docs for query engine
    path_handbook_2023 = '/mnt/efs/data/AIEresearch/demo_medicare_handbook/data/Medicare-and-You.2023 National Version.pdf'
    path_handbook_2024 = '/mnt/efs/data/AIEresearch/demo_medicare_handbook/data/10050-Medicare-and-You.pdf'
    lst_docs = [path_handbook_2024]

    # Create query engine
    ## Turn off streaming if TruLens being used
    streaming = False if args.trulens else True
    query_engine = load_llama2_rag(model, tokenizer, lst_docs, streaming)
    
    # Run only if GPUs are available
    if not torch.cuda.is_available():
        description += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"
    
    # Start tru object
    if args.trulens:
        tru = Tru()
        tru.reset_database()

        # Initialize config parser
        config = configparser.ConfigParser()
        config.read("/mnt/efs/data/AIEresearch/config.ini")
        # Set the OpenAI authorization token 
        openai_key = config['openai']['api_key']
        os.environ['OPENAI_API_KEY'] = openai_key

        # Initialize groundedness
        openai = OpenAI()
        grounded = Groundedness(groundedness_provider=openai)

        # Define a groundedness feedback function
        f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons) \
            .on(TruLlama.select_source_nodes().node.text.collect()) \
            .on_output() \
            .aggregate(grounded.grounded_statements_aggregator)

        # Question/answer relevance between overall question and answer.
        f_qa_relevance = Feedback(openai.relevance).on_input_output()

        # Question/statement relevance between question and each context chunk.
        f_qs_relevance = Feedback(openai.qs_relevance).on_input().on(
                            TruLlama.select_source_nodes().node.text).aggregate(np.mean)

        # Set up query engine recorder
        tru_query_engine_recorder = TruLlama(query_engine,
                                            app_id='LlamaIndex_App',
                                            feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance])     
        # Start TruLens dashboard
        tru.run_dashboard() 

    # Set generate function
    def generate(
        message: str, #DISSECT: User input
        chat_history: list[tuple[str, str]],
        system_prompt: str, #DISSECT: Not being used. Not sure how to use. 
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
    ) -> Iterator[str]:

        #DISSECT > User input
        print('\n', '*'*100, '\n message:', message, '\n', '*'*100)

        conversation = []
        
        if system_prompt:
            #DISSECT > Not used
            #DISSECT > How can this be used?
            print('\n', '*'*100, '\n system_prompt:', system_prompt, '\n', '*'*100)
            
            conversation.append({"role": "system", "content": system_prompt})
        
        for user, assistant in chat_history:
            #DISSECT > Appears after user inputs SECOND message
            #DISSECT > [['This is my first message to you. ', " Hello! *smiling* It's nice to meet you, and thank you for reaching out. I'm here to help answer any questions or provide assistance you may need. How can I assist you today?"]]
            #DISSECT > When does this get updated?
            #DISSECT > How is the udpate passed into follow-on invocations of the generate function?
            print('\n', '*'*100, '\n chat_history:', chat_history, '\n', '*'*100)

            conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

            #DISSECT > Appears after user inputs SECOND message
            #DISSECT > conversation_chathistory: [{'role': 'user', 'content': 'This is my first message to you. '}, {'role': 'assistant', 'content': " Hello! *smiling* It's nice to meet you, and thank you for reaching out. I'm here to help answer any questions or provide assistance you may need. How can I assist you today?"}] 
            print('\n', '*'*100, '\n conversation_chathistory:', conversation, '\n', '*'*100)

        conversation.append({"role": "user", "content": message})

        #DISSECT > Used before a first time conversation_chathistory likely becuase chat_history is empty at first
        #DISSECT > Fist output: [{'role': 'user', 'content': 'This is my first message to you. '}] 
        #DISSECT > Second output: conversation_postchathistory: [{'role': 'user', 'content': 'This is my first message to you. '}, {'role': 'assistant', 'content': " Hello! *smiling* It's nice to meet you, and thank you for reaching out. I'm here to help answer any questions or provide assistance you may need. How can I assist you today?"}, {'role': 'user', 'content': 'This is my second message to you. '}] 
        print('\n', '*'*100, '\n conversation_postchathistory:', conversation, '\n', '*'*100)

        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
            gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
        input_ids = input_ids.to(model.device) #MODEL leave as is?

        streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            {"input_ids": input_ids},
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
            repetition_penalty=repetition_penalty,
        )
        t = Thread(target=model.generate, kwargs=generate_kwargs) #MODEL
        # t = Thread(target=query_engine.query, kwargs=generate_kwargs) #MODEL
        t.start()

        # ADDED TO CAPTURE CHAT LOGS
        with open(log_file_path, 'a') as f:
            f.write(f"QUERY: {message}\n")

        outputs = []

        for text in streamer:
            outputs.append(text)
            yield "".join(outputs)

        # ADDED TO CAPTURE CHAT LOGS
        with open(log_file_path, 'a') as f:
            f.write(f"RESPONSE: {''.join(outputs)}\n\n")    
    
    # Load questions from JSON file and iterate over them
    with open('medicare_demo_questions.json', 'r') as file:
        questions = json.load(file)

    for question in questions:
        question_text = question["Question"]
        chat_history = []
        response_generator = query_index(question_text, chat_history)

        for response in response_generator:
            print("Question:", question_text)
            print("Response:", response)
            print("\n" + "-"*50 + "\n")
      
    # Set up chat interface
    chat_interface = gr.ChatInterface(
        # fn=generate,
        fn=query_index,
        # additional_inputs=[
            # gr.Textbox(label="System prompt", lines=6),
            # gr.Slider(
            #     label="Max new tokens",
            #     minimum=1,
            #     maximum=MAX_MAX_NEW_TOKENS,
            #     step=1,
            #     value=DEFAULT_MAX_NEW_TOKENS,
            # ),
            # gr.Slider(
            #     label="Temperature",
            #     minimum=0.1,
            #     maximum=4.0,
            #     step=0.1,
            #     value=0.6,
            # ),
            # gr.Slider(
            #     label="Top-p (nucleus sampling)",
            #     minimum=0.05,
            #     maximum=1.0,
            #     step=0.05,
            #     value=0.9,
            # ),
            # gr.Slider(
            #     label="Top-k",
            #     minimum=1,
            #     maximum=1000,
            #     step=1,
            #     value=50,
            # ),
            # gr.Slider(
            #     label="Repetition penalty",
            #     minimum=1.0,
            #     maximum=2.0,
            #     step=0.05,
            #     value=1.2,
            # ),
        # ],
        stop_btn=None,
        examples=[
            ["Can you please tell me about Medicare?"],
            ["What is the difference between Medicare and Medicaid?"],
            ["How many parts does Medicare have?"]
        ],
    )

    # Render demo
    with gr.Blocks(css="style.css") as demo:
        gr.Markdown(description)
        chat_interface.render()
   
    # Start demo
    demo.queue(max_size=20).launch()