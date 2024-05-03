'''
https://huggingface.co/docs/transformers/main/en/model_doc/mistral#transformers.MistralModel.forward.attention_mask
'''
# Added as per https://stackoverflow.com/questions/69426453/declaration-of-list-of-type-python
from __future__ import annotations

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
from threading import Thread
from typing import Iterator

import torch
import gradio as gr
from transformers import TextIteratorStreamer


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
    # {model}

    [{model}](https://huggingface.co/mistralai/{model}) by Mistral, a genereative model with 7B parameters for text generation. 
    Note that this mode does not have any moderation mechanisms.
    """
    return DESCRIPTION


if __name__ == "__main__":  
    # Run only if GPUs are available
    if not torch.cuda.is_available():
        print("Running on CPU 🥶 This demo does not work on CPU.")
        # sys.exit()
   
    # Clear GPUs if available
    aie_helper.report_gpu()

    # Create parser
    parser = argparse.ArgumentParser(prog='app.py',
                                     description='Run Mistral7B model on browser.')
    # Add argument for size
    parser.add_argument('-m', '--mixture_of_expertts', required=False, action='store_true',
                        help='Add this flag to use mixture of experts Mixtral-8x7B-v0.1.')
    # Add argument for name to use in logs
    parser.add_argument('-l', '--log_name', type=str, required=True,
                        help='String name to be used in log file.')
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
        model, tokenizer = aie_helper.load_mistral8x7b_model()
    else:
        model, tokenizer = aie_helper.load_mistral7b_model()

    # Set generate function
    def generate(
        message: str,
        chat_history: list[tuple[str, str]],
        max_new_tokens: int = 2048,
        temperature: float = 0.3,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> Iterator[str]:
        # Start the timer
        start = time.time()
        
        # Set log_idx
        log_idx = len(chat_history)
        
        # Set temp
        if temperature < 1e-2:
            temperature = 1e-2

        conversation = []

        for user, assistant in chat_history:
            conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
        conversation.append({"role": "user", "content": message})

        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
        
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
            gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
        input_ids = input_ids.to(model.device)

        streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            {"input_ids": input_ids},
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            num_beams=1,
            repetition_penalty=repetition_penalty,
        )
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        # Add to log
        with open(log_file_path, 'w') as f:
            log_dict[log_idx]['query'] = message
            f.write(json.dumps(log_dict))

        outputs = []
        for text in streamer:
            outputs.append(text)
            yield "".join(outputs)
        
        # Add to log
        with open(log_file_path, 'w') as f:
            log_dict[log_idx]['response'] = ''.join([o for o in outputs])
            log_dict[log_idx]['time'] = str(timedelta(seconds=time.time()-start))
            f.write(json.dumps(log_dict))

    # Set up chat interface 
    chat_interface = gr.ChatInterface(
        fn=generate,
        additional_inputs=[
            gr.Slider(
                label="Max new tokens",
                minimum=1,
                maximum=MAX_MAX_NEW_TOKENS,
                step=1,
                value=DEFAULT_MAX_NEW_TOKENS,
            ),
            gr.Slider(
                label="Temperature",
                minimum=0.1,
                maximum=4.0,
                step=0.1,
                value=0.6,
            ),
            gr.Slider(
                label="Top-p (nucleus sampling)",
                minimum=0.05,
                maximum=1.0,
                step=0.05,
                value=0.9,
            ),
            gr.Slider(
                label="Repetition penalty",
                minimum=1.0,
                maximum=2.0,
                step=0.05,
                value=1.2,
            ),
        ],
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
    demo.queue(max_size=20).launch(server_port=7860)