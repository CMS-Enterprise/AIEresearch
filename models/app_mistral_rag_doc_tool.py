'''
https://huggingface.co/docs/transformers/main/en/model_doc/mistral#transformers.MistralModel.forward.attention_mask
'''
# # Added as per https://stackoverflow.com/questions/69426453/declaration-of-list-of-type-python
# from __future__ import annotations

# Import custom functions
import sys
sys.path.insert(1, '~/models/aie_helper_functions.py') #Ensure this is the right path in your env!
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
from llama_index import SimpleDirectoryReader

# TrueLens
from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI
import numpy as np

# Doc Upload Tool
from llama_index import SimpleDirectoryReader

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
    # return DESCRIPTION
    return 'Upload one or several files, then press "Click to Upload File(s) to use a chatbot that references the uploaded documents for answers!'


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
    log_file_path_save = '~/aie_demo_playground/chat_logs' #Ensure this is the right path in your env!
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

    lst_docs = []
    # Set function for handling uploaded files
    def upload_file(files):
        # Get file paths
        global lst_docs
        lst_docs = [file.name for file in files]
        return lst_docs

    # Create engine
    def create_engine(progress=gr.Progress()):
        # Create query engine
        ## Turn off streaming if TruLens being used
        streaming = False #if args.trulens else True
        global engine
        # Load engine based on engine type
        if args.engine_type==0:
            engine = aie_helper.load_mistral7b_query_engine(model, tokenizer, lst_docs, streaming)
        elif args.engine_type==1:
            engine = aie_helper.load_mistral7b_CondenseQuestionChatEngine(model, tokenizer, lst_docs, streaming)

        # Set up TruLens if necessary
        if args.trulens:
            print('SETTING UP TRULENS')
            # Start tru object
            tru = Tru(database_url="sqlite:///doc_upload.sqlite")
            tru.reset_database()

            # Set your OpenAI authorization token 
            os.environ['OPENAI_API_KEY'] = 'INSERT YOUR OPENAI API KEY HERE'

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
            global tru_query_engine_recorder
            tru_query_engine_recorder = TruLlama(engine,
                                                app_id='LlamaIndex_App',
                                                feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance])     
            # Start TruLens dashboard
            tru.run_dashboard(port='8089') 

        imgs = [None] * 24
        for img in progress.tqdm(imgs, desc="Loading..."):
            time.sleep(0.1)
        return ["Loaded"] * 2

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

        # Stream response to GUI        
        outputs = []
        resp = ''
        # Set response object based on if trulens is working
        resp_obj = response.response #if args.trulens else response.response_gen
        for text in resp_obj:
            outputs.append(text)
            resp += text
        # Add to chat history 
        chat_history.append((message, resp))
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
        return '', chat_history
                 
    # Set up demo
    # https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks
    with gr.Blocks(css="style.css") as demo:
        gr.Markdown(description)
        
        # Set layout items 
        ## File upload
        upload_button = gr.UploadButton("Click to Upload File(s)", 
                                        file_types=["file"], 
                                        file_count="multiple")
        file_output   = gr.File()
        upload_button.upload(upload_file, upload_button, file_output)
        
        ## Chatbot
        launch_model = gr.Button("Click to Launch Model with Files")
        launch_model.click(fn=create_engine) #gr.Progress(), 
        chatbot = gr.Chatbot([], 
                             elem_id="chatbot", 
                             label='Chatbox')
        text_box = gr.Textbox(label= "Question",
                              lines=2,
                              placeholder="Type a message...")
       ## Chatbot buttons
        with gr.Row():
            with gr.Column(scale=0.5):
                submit_btn = gr.Button('Submit',
                                       variant='primary', 
                                       size='sm')
            with gr.Column(scale=0.5):
                clear_chat_btn = gr.Button('Clear Chat',
                                           variant='stop',
                                           size='sm')
        
        # Set button functionality
        submit_btn.click(fn=query_index, 
                         inputs=[text_box, chatbot], 
                         outputs=[text_box, chatbot])
        
        clear_chat_btn.click(fn=lambda: None, 
                             inputs=None, 
                             outputs=chatbot, 
                             queue=False)
        

    # Start the demo
    demo.queue(max_size=20).launch(server_port=7868)