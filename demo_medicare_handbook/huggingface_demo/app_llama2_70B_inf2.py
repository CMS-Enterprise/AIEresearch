'''
Use .venv_dev310
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

import torch
import gradio as gr
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers_neuronx.llama.model import LlamaForSampling

LICENSE = """
<p/>

---
As a derivate work of [Llama-2-70b-chat](https://huggingface.co/meta-llama/Llama-2-70b-chat) by Meta using llama_cpp,
this demo is governed by the original [license](https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat/blob/main/LICENSE.txt) and [acceptable use policy](https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat/blob/main/USE_POLICY.md).
"""

# Create description
def create_description():
    '''
    Returns text description used in model page for model being used. 

    Args:
        None

    Returns:
        DESCRIPTION (str): Text used in model page. 
    '''
    DESCRIPTION = f"""\
    # Llama-2 70B Chat through llama_cpp

    This Space demonstrates model [Llama-2-70b-chat](https://huggingface.co/meta-llama/Llama-2-70b-chat) by Meta, a Llama 2 model with 70B parameters fine-tuned for chat instructions 
    used through llama_cpp. 
    🔎 For more details about the Llama 2 family of models and how to use them with `transformers`, take a look [at this Hugging Face blog post](https://huggingface.co/blog/llama2).

    """
    return DESCRIPTION


if __name__ == "__main__":  
    # Create parser
    parser = argparse.ArgumentParser(prog='app.py',
                                     description='Run Hugging Face Llama2 model on browser.')
    # Add argument for name to use in logs
    parser.add_argument('-l', '--log_name', type=str, required=True,
                        help='String name to be used in log file.')
    # Parser args
    args = parser.parse_args()

    # Create description
    description = create_description()

    # Set up log file
    log_file_path_save = '/mnt/efs/data/AIEresearch/demo_medicare_handbook/chat_logs'
    # log_file_name = f"mixtral8x7B_Inf2_{args.log_name}" if args.mixture_of_expertts else f"mistral7B_inf2_{args.log_name}"
    log_file_name = f"llama2_70B_inf2_{args.log_name}"
    log_file_path = aie_helper.set_up_log_file(log_file_name, log_file_path_save)
    # Set log_dict to contain info for json
    log_dict = defaultdict(dict)
    
    os.environ['NEURON_CC_FLAGS'] = '--enable-mixed-precision-accumulation'

    # Load meta-llama/Llama-2-70b to the NeuronCores with 8-way tensor parallelism and run compilation
    neuron_model = LlamaForSampling.from_pretrained(
        '/mnt/efs/data/saved_models/Llama-2-70b-split',  # Should reference the split checkpoint produced by "save_pretrained_split"
        batch_size=1,           # Batch size must be determined prior to inference time.
        tp_degree=24,           # Controls the number of NeuronCores to execute on. Change to 32 for trn1.32xlarge
        amp='f16',              # This automatically casts the weights to the specified dtype.
    )
    neuron_model.to_neuron()

    # Get a tokenizer and exaple input
    # tokenizer = AutoTokenizer.from_pretrained('upstage/Llama-2-70b-instruct') 
    tokenizer = AutoTokenizer.from_pretrained('/mnt/efs/data/saved_models/Llama-2-70b-chat-hf/tokenizer') 
    print('tokenizer initiated\n')

    # Set inference function
    def inference(message, chat_history):
        # Start the timer
        start = time.time()

        # Set log_idx
        log_idx = len(chat_history)
        
        # Add to log
        with open(log_file_path, 'w') as f:
            log_dict[log_idx]['query'] = message
            f.write(json.dumps(log_dict))

        # Encode the input
        input_ids = tokenizer.encode(message, return_tensors="pt")
        
        # Run inference
        with torch.inference_mode():
            generated_sequences = neuron_model.sample(input_ids, sequence_length=2048, top_k=50)

        # Turn tokens to words
        generated_sequences = [tokenizer.decode(seq) for seq in generated_sequences]

        # Set the splitter
        splitter = '\n\n'
        if '\nAnswer: ' in generated_sequences[0]:
            splitter = '\nAnswer: '
        elif '\nA: ' in generated_sequences[0]:
            splitter = '\nA: '
        
        outputs = []
        for tok in generated_sequences[0].split(splitter)[1:]:
            tok = tok.replace('</s>', '')
            outputs.append(tok)
            yield "".join(outputs)
        
        # Add to log
        with open(log_file_path, 'w') as f:
            log_dict[log_idx]['response'] = ''.join([o for o in outputs])
            log_dict[log_idx]['time'] = str(timedelta(seconds=time.time()-start))
            f.write(json.dumps(log_dict))

    # Set up chat interface 
    chat_interface = gr.ChatInterface(
        fn=inference,
        stop_btn=None,
        examples=[
                ["Can you please tell me about Medicare?"],
                ["What is the difference between Medicare and Medicaid?"],
                ["How many parts does Medicare have?"]
        ],
    )

    # Set up demo
    with gr.Blocks(css="style.css") as demo:
        gr.Markdown(description)
        chat_interface.render()
        gr.Markdown(LICENSE)

    # Start the demo
    demo.queue(max_size=20).launch()