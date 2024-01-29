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

from transformers_neuronx import constants
from transformers_neuronx.mistral.model import MistralForSampling
from transformers_neuronx.module import save_pretrained_split
from transformers_neuronx.config import NeuronConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    # {model} on AWS Inferentia2 Chip

    [{model}](https://huggingface.co/mistralai/{model}) by Mistral, a genereative model with 7B parameters for text generation. 
    Note that this mode does not have any moderation mechanisms.
    """
    return DESCRIPTION


if __name__ == "__main__":  
    # Create parser
    parser = argparse.ArgumentParser(prog='app.py',
                                     description='Run Mistral7B model on browser.')
    # Add argument for name to use in logs
    parser.add_argument('-l', '--log_name', type=str, required=True,
                        help='String name to be used in log file.')
    # Parser args
    args = parser.parse_args()

    # Create description
    description = create_description()

    # Set up log file
    log_file_path_save = '/mnt/efs/data/AIEresearch/demo_medicare_handbook/chat_logs'
    log_file_name = f"mistral7B_inf2_{args.log_name}"
    log_file_path = aie_helper.set_up_log_file(log_file_name, log_file_path_save)
    # Set log_dict to contain info for json
    log_dict = defaultdict(dict)
    
    # Load and save the CPU model with bfloat16 casting
    model_cpu = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
    # model_cpu = AutoModelForCausalLM.from_pretrained('/mistralai/Mistral-7B-Instruct-v0.1-split')  # CHANGE THIS TO USE PRESAVED MODEL@
    # save_pretrained_split(model_cpu, 'mistralai/Mistral-7B-Instruct-v0.1-split')

    # Set sharding strategy for GQA to be shard over heads
    neuron_config = NeuronConfig(
        grouped_query_attention=constants.GQA.SHARD_OVER_HEADS
    )
    print('neuron_config initiated\n')

    # Create and compile the Neuron model
    model_neuron = MistralForSampling.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1-split', batch_size=1, \
        tp_degree=2, n_positions=256, amp='bf16', neuron_config=neuron_config)
    model_neuron.to_neuron()
    print('neuron model compiled\n')

    # Get a tokenizer and exaple input
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
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
        encoded_input = tokenizer(f"[INST]  {message} [/INST]", return_tensors='pt')
        
        # Run inference
        with torch.inference_mode():
            generated_sequences = model_neuron.sample(encoded_input.input_ids, sequence_length=256, start_ids=None)

        outputs = []
        for tok in generated_sequences:
            output = tokenizer.decode(tok)
            if '[/INST]' in output:
                outputs.append(output.split('[/INST]')[-1].split('</s>')[0])
            else:
                outputs.append(output)
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