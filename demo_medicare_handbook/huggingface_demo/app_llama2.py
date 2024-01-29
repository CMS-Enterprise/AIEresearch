# Noblis ESI change for AIE
'''
Instuction tuning guide: https://www.philschmid.de/instruction-tune-llama-2


'''
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
from threading import Thread
from typing import Iterator

import gradio as gr
import torch
from transformers import TextIteratorStreamer


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
    # Parser args
    args = parser.parse_args()

    # Create desscription 
    description = create_description(args.size)

    # Set up log file
    log_file_path_save = '/mnt/efs/data/AIEresearch/demo_medicare_handbook/chat_logs'
    log_file_name = f"llama2_{args.size}B_{args.log_name}"
    log_file_path = aie_helper.set_up_log_file(log_file_name, log_file_path_save)
    # Set log_dict to contain info for json
    log_dict = defaultdict(dict)
    
    # Create model and tokenizer objects
    model, tokenizer = aie_helper.load_llama2_model(args.size)
    # Adjust tokenizer for use here
    tokenizer.use_default_system_prompt = False
    
    # Run only if GPUs are available
    if not torch.cuda.is_available():
        description += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"
    
    # Set generate function
    def generate(
        message: str,
        chat_history: list[tuple[str, str]],
        system_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
    ) -> Iterator[str]:
        # Start the timer
        start = time.time()
        
        # Set log_idx
        log_idx = len(chat_history)
                
        conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
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
            top_k=top_k,
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
            gr.Textbox(label="System prompt", lines=6),
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
                label="Top-k",
                minimum=1,
                maximum=1000,
                step=1,
                value=50,
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

    # Render demo
    with gr.Blocks(css="style.css") as demo:
        gr.Markdown(description)
        chat_interface.render()
   
    # Start demo (queue is slower)
    demo.queue(max_size=20).launch()
    # demo.launch()