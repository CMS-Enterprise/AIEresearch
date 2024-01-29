'''
https://huggingface.co/docs/transformers/main/en/model_doc/mistral#transformers.MistralModel.forward.attention_mask
'''
# Added as per https://stackoverflow.com/questions/69426453/declaration-of-list-of-type-python
from __future__ import annotations

# Import custom functions
import sys
sys.path.insert(1, '/mnt/efs/data/AIEresearch/')
import aie_helper_functions as aie_helper

# Imports
import os
import json
import time
import argparse
from datetime import timedelta
from collections import defaultdict
from typing import Iterator

import torch
import gradio as gr
# from transformers import TextIteratorStreamer

# TrueLens
import configparser
from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI
import numpy as np


# Global Variables
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 2048
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

LICENSE = """
<p/>

---
Governed by [Appache License 2.0](https://github.com/openstack/mistral/blob/master/LICENSE).
"""

# Create description
def create_description(moe=False):
    '''
    Returns text description used in model page for model being used. 

    Args:
        moe (bool): True to use mixture of experts model Mixtral-8x7B-v0.1. 
            False to use Mistral-7B-Instruct-v0.2.

    Returns:
        DESCRIPTION (str): Text used in model page. 
    '''
    model = 'Mixtral-8x7B-v0.1' if moe else 'Mistral-7B-Instruct-v0.2'
    DESCRIPTION = f"""\
    # {model} RAG

    [{model}](https://huggingface.co/mistralai/{model}) by Mistral, a genereative model with 7B parameters for text generation. 
    Note that this mode does not have any moderation mechanisms.
    """
    return DESCRIPTION


if __name__ == "__main__":  
    # Run only if GPUs are available
    if not torch.cuda.is_available():
        print("Running on CPU 🥶 This demo does not work on CPU.")
        sys.exit()
   
    # Clear GPUs if available
    aie_helper.report_gpu()

    # Create parser
    parser = argparse.ArgumentParser(prog='app.py',
                                     description='Run Mistral7B model on browser.')
    # Add argument for size
    parser.add_argument('-m', '--mixture_of_expertts', required=False, action='store_true',
                        help='Add this flag to use mixture of experts Mixtral-8x7B-v0.1.')
    # Add argument for query type
    parser.add_argument('-e', '--engine_type', type=int, required=False, default=0, choices=[0, 1], 
                        help="0 for query_engine, 1 for CondenseQuestionChatEngine")
    # Add argument for name to use in logs
    parser.add_argument('-l', '--log_name', type=str, required=True,
                        help='String name to be used in log file.')
    # Add argument for using TruLens
    parser.add_argument('-t', '--trulens', required=False, action='store_true',
                        help='Add this flag to use TruLens.')
    # Parser args
    args = parser.parse_args()

    # Create description
    description = create_description(args.mixture_of_expertts)

    # Set up log file
    log_file_path_save = '/mnt/efs/data/AIEresearch/demo_medicare_handbook/chat_logs'
    log_file_name = f"mixtral8x7B_{args.log_name}" if args.mixture_of_expertts else f"mistral7B_{args.log_name}"
    log_file_path = aie_helper.set_up_log_file(log_file_name, log_file_path_save)
    # Set log_dict to contain info for json
    log_dict = defaultdict(dict)
    
    # Create model and tokenizer objects
    if args.mixture_of_expertts:
        print("This model is not yet supported.")
        sys.exit()
    else:
        model, tokenizer = aie_helper.load_mistral7b_model()

    # Set up docs for query engine
    path_handbook_2023 = '/mnt/efs/data/AIEresearch/demo_medicare_handbook/data/Medicare-and-You.2023 National Version.pdf'
    path_handbook_2024 = '/mnt/efs/data/AIEresearch/demo_medicare_handbook/data/10050-Medicare-and-You.pdf'
    lst_docs = [path_handbook_2024]

    # Create query engine
    ## Turn off streaming if TruLens being used
    streaming = False if args.trulens else True
    # Load engine based on engine type
    if args.engine_type==0:
        engine = aie_helper.load_mistral7b_query_engine(model, tokenizer, lst_docs, streaming)
    elif args.engine_type==1:
        engine = aie_helper.load_mistral7b_CondenseQuestionChatEngine(model, tokenizer, lst_docs, streaming)
        # DOES THIS WORK WITH TRULENS?

    # Set up TruLens if necessary
    if args.trulens:
        # Start tru object
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
        tru_query_engine_recorder = TruLlama(engine,
                                             app_id='LlamaIndex_App',
                                             feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance])     
        # Start TruLens dashboard
        tru.run_dashboard(port='8080') 

    # query function using rag model
    def query_index(message, chat_history):
        # Start the timer
        start = time.time()
        
        # Set log_idx
        log_idx = len(chat_history)
    # async def query_index(message, chat_history): # Approach to get async to work 
        # Add to log
        with open(log_file_path, 'w') as f:
            log_dict[log_idx]['query'] = message
            f.write(json.dumps(log_dict))

        # Set the streaming argument based on whether TruLens in active
        if args.trulens:
            # Set up TruLens context manager
            with tru_query_engine_recorder as recording:
                # Get non-streaming response
                response = engine.query(message) 
        else:
            # Get streaming response based on model type
            if args.engine_type==0:
                response = engine.query(message)
            elif args.engine_type==1:
                # Assume no streaming for now
                response = engine.chat(message)
                # If streaming
                # response = engine.stream_chat(message)

        # Stream response to GUI        
        outputs = []
        # Set response object based on if trulens is working
        resp_obj = response.response if args.trulens else response.response_gen
        for text in resp_obj:
            outputs.append(text)
            yield "".join(outputs)
        # Add to log
        with open(log_file_path, 'w') as f:
            log_dict[log_idx]['response'] = ''.join([o for o in outputs])
            f.write(json.dumps(log_dict))

        # Add to log
        with open(log_file_path, 'a') as f:
            # Aggregate source data into source_lst 
            source_lst = []
            # Iterate over source nodes
            for source_node in response.source_nodes:
                # Create a dict to contain source data
                source_dict = {}
                # Add metadata
                source_dict['metadata'] = source_node.metadata
                # Add content
                source_dict['content'] = source_node.get_text()
                # Add to source_lst
                source_lst.append(source_dict)
            # Add to logs
            with open(log_file_path, 'w') as f:
                log_dict[log_idx]['source'] = source_lst
                log_dict[log_idx]['time'] = str(timedelta(seconds=time.time()-start))
                f.write(json.dumps(log_dict))

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
        fn=query_index,
        stop_btn=None,
        examples=[
            ["Can you please tell me about Medicare?"],
            ["What is the difference between Medicare and Medicaid?"],
            ["How many parts does Medicare have?"]
        ],
    )

    # Set up demo
    with gr.Blocks(css="style.css") as demo:
        # gr.Markdown(description)
        chat_interface.render()
        # gr.Markdown(LICENSE)

    # Start the demo
    demo.queue(max_size=20).launch()