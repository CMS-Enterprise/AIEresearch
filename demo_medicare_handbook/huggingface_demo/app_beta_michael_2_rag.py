# Noblis ESI change for AIE
'''
Instuction tuning guide: https://www.philschmid.de/instruction-tune-llama-2

TODO:
* Load saved query_engine?
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
from typing import Iterator

import gradio as gr
import torch
from transformers import TextIteratorStreamer

# TrueLens
import configparser
from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI
import numpy as np

# Test Imports
import json

# Global variables
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

if __name__ == "__main__":
    # Run only if GPUs are available
    if not torch.cuda.is_available():
        #description += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"
        sys.exit()
    
    # Clear GPUs if available
    aie_helper.report_gpu()
    
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
    log_file_name = f"llama2_{args.size}B_{args.log_name}"
    log_file_path = aie_helper.set_up_log_file(log_file_name, log_file_path_save)
    
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
    query_engine = aie_helper.load_llama2_rag(model, tokenizer, lst_docs, streaming)
    
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
        tru_query_engine_recorder = TruLlama(query_engine,
                                            app_id='LlamaIndex_App',
                                            feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance])     
        # Start TruLens dashboard
        tru.run_dashboard()      
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
    with open('medicare_demo_questions_test.json', 'r') as file:
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

    # Render demo
    with gr.Blocks(css="style.css") as demo:
        gr.Markdown(description)
        chat_interface.render()
        gr.Markdown(LICENSE)
   
    # Start demo
    demo.queue(max_size=20).launch()