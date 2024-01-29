'''
Common functions used by the AIE team stored in one place 
for convinient use and reuse, and to ensure that changes
are consistent and persistent.

AIE Team 
Last updated: December 2023
'''

# Imports
import os
import gc
import torch
from datetime import datetime
from torch import bfloat16

# Model quantization
from transformers import (BitsAndBytesConfig, 
                          AutoModelForCausalLM, 
                          AutoTokenizer)

# RAG
from llama_index import (SimpleDirectoryReader, 
                         Document, 
                         ServiceContext, 
                         VectorStoreIndex)
from llama_index.llms import (HuggingFaceLLM,
                              ChatMessage, 
                              MessageRole)
from llama_index.prompts import PromptTemplate
from llama_index.embeddings import LangchainEmbedding
from llama_index.prompts import PromptTemplate #Please use langchain.prompts.PromptTemplate instead.
from llama_index.chat_engine.condense_question import CondenseQuestionChatEngine

from langchain.embeddings.huggingface import HuggingFaceEmbeddings


# GLOBAL VARIABLES

# Set quantization configuration
# See https://huggingface.co/blog/4bit-transformers-bitsandbytes
NF4_CONFIG = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=bfloat16)



# HELPER FUNCTIONS

# Set up log file
def set_up_log_file(log_name, path_save):
    '''
    Set name and path for logfile. Return path. 

    Args:
        log_name (str): Name to be used in log file. 
        path_save (str): Path to save log files. 

    Returns:
        log_file_path (str): Path for log file. 
    '''
    file_name = f"{log_name}_{datetime.now()}.json"
    file_name = file_name.replace(' ', '_')
    log_file_path = os.path.join(path_save, file_name)
    return log_file_path


# Added to try to address GPU memory issues
# Garbage collection and memory management
def report_gpu():
   print('Clear GPUs')
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()
   print()


# Load Llama2 model saved to disk
def load_llama2_model(size, 
                      quantization=True,
                      save_dir='/mnt/efs/data/saved_models'):
    '''
    Loads Llama2 model saved in mnt/efs/data/saved_models.

    Args:
        size (int): 7 for 7B, 13 for 13B, and 70 for 70B.
        quantization (bool): True to use nf4_config. 
        save_dir (str): Path where models are saved.

    Returns:
        model (transformers.models.llama.modeling_llama.LlamaForCausalLM): Model.
        tokenizer (transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast): 
            Tokenizer.
    '''
    # Set model name 
    model_name = f"Llama-2-{size}b-chat-hf"

    if torch.cuda.is_available():
        # Set model paths 
        path_model_save           = os.path.join(save_dir, model_name)
        path_model_save_model     = os.path.join(path_model_save, 'model')
        path_model_save_tokenizer = os.path.join(path_model_save, 'tokenizer')
        # Load model based on quantization
        if quantization:
            model = AutoModelForCausalLM.from_pretrained(path_model_save_model, quantization_config=NF4_CONFIG, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(path_model_save_model, load_in_4bit=False, device_map='auto')
        # Load tokernizer
        tokenizer = AutoTokenizer.from_pretrained(path_model_save_tokenizer) 
        # Return 
        return model, tokenizer


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
    documents = SimpleDirectoryReader(input_files=lst_docs).load_data()
    
    # Print document stats
    print()
    print(f"Documents type: {type(documents)}")
    print(f"Number of pages: {len(documents)}")
    print(f"Sub-document type: {type(documents[0])}")
    print(f"{documents[0]}\n")
    print()
    
    # # Merge documents into one
    # document = Document(text="\n\n".join([doc.text for doc in documents]))

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

    # Create a HF LLM using the llama index wrapper 
    llm = HuggingFaceLLM(context_window=4096,
                        max_new_tokens=2048, #256,
                        system_prompt=system_prompt,
                        query_wrapper_prompt=PromptTemplate("<s> [INST] {query_str} [/INST] "),
                        # generate_kwargs={"temperature": 0.3, "do_sample": True},
                        # tokenizer_kwargs={"max_length": 4096},
                        model_kwargs={"quantization_config": NF4_CONFIG}, #Is this necessary if the model is already quantized?
                        device_map="auto",
                        model=model,
                        tokenizer=tokenizer)

    # Create and dl embeddings instance  
    embeddings = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
    print('embeddings loaded.\n')

    # Create new service context instance
    service_context = ServiceContext.from_defaults(chunk_size=1024,
                                                   llm=llm,
                                                   embed_model=embeddings)
    # service_context = ServiceContext.from_defaults(llm=llm, 
    #                                                embed_model="local:/mnt/efs/data/saved_models/BAAI/bge-small-en-v1.5/model/")
    print('service_context created.\n')

    # Create the index
    index = VectorStoreIndex.from_documents(documents, #[document],
                                            service_context=service_context)
    print('index created.\n')

    # Create the query engine
    query_engine = index.as_query_engine(streaming=streaming)
    print('query_engine loaded.\n')

    # Return 
    return query_engine


# Load Mistral7B model saved to disk
def load_mistral7b_model(quantization=True,
                         save_dir='/mnt/efs/data/saved_models'):
    '''
    Loads Mistral-7B-Instruct-v0.2 model saved in mnt/efs/data/saved_models.

    Args:
        quantization (bool): True to use nf4_config. 
        save_dir (str): Path where models are saved.

    Returns:
        model (transformers.models.llama.modeling_llama.LlamaForCausalLM): Model.
        tokenizer (transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast): 
            Tokenizer.
    '''
    # Set model name 
    model_name = 'Mistral-7B-Instruct-v0.2'

    if torch.cuda.is_available():
        # Set model paths 
        path_model_save           = os.path.join(save_dir, model_name)
        path_model_save_model     = os.path.join(path_model_save, 'model')
        path_model_save_tokenizer = os.path.join(path_model_save, 'tokenizer')
        # Load model based on quantization
        if quantization:
            model = AutoModelForCausalLM.from_pretrained(path_model_save_model, quantization_config=NF4_CONFIG, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(path_model_save_model, load_in_4bit=False, device_map='auto')
        # Load tokernizer
        tokenizer = AutoTokenizer.from_pretrained(path_model_save_tokenizer) 
        # Return 
        return model, tokenizer


def load_mistral7b_query_index(model, tokenizer, lst_docs, streaming=True):
    '''
    Return query_engine for given documents. 

    Args:
        model (transformers.models.mistral.modeling_mistral.MistralForCausalLM): Mistral 7B model (any size).
        tokenizer (transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast): Llama2 tokenizer.
        lst_doct (list): List of documents to be used in RAG model. 
        streaming (bool): True to set a streaming model, False otherwise. Defaults to True. 

    Returns:
        query_engine (type)
    
    TODO:
        Make embedding model and input. 
    '''
    # Read in PDF(s) into a llamaindex document object
    documents = SimpleDirectoryReader(input_files=lst_docs).load_data()
    
    # Print document stats
    print()
    print(f"Documents type: {type(documents)}")
    print(f"Number of pages: {len(documents)}")
    print(f"Sub-document type: {type(documents[0])}")
    print(f"{documents[0]}\n")
    print()
    
    # # Merge documents into one
    # document = Document(text="\n\n".join([doc.text for doc in documents]))

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

    # # Create a HF LLM using the index wrapper 
    # llm = HuggingFaceLLM(context_window=4096,
    #                      max_new_tokens=2048, #256,
    #                      system_prompt=system_prompt,
    #                      #  query_wrapper_prompt=query_wrapper_prompt,
    #                      model=model,
    #                      tokenizer=tokenizer, 
    #                      device_map='auto')
    # Create a HF LLM using the llama index wrapper 
    llm = HuggingFaceLLM(context_window=4096,
                        max_new_tokens=2048, #256,
                        system_prompt=system_prompt,
                        query_wrapper_prompt=PromptTemplate("<s>[INST] {query_str} [/INST] </s>\n"),
                        #  generate_kwargs={"temperature": 0.3, "do_sample": True},
                        generate_kwargs={"temperature": 0.2, "top_k": 5, "top_p": 0.95},
                        tokenizer_kwargs={"max_length": 4096},
                        model_kwargs={"quantization_config": NF4_CONFIG},
                        device_map="auto",
                        model=model,
                        tokenizer=tokenizer)

    # Create and dl embeddings instance  
    embeddings = LangchainEmbedding(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    print('embeddings loaded.\n')

    # Create new service context instance
    service_context = ServiceContext.from_defaults(chunk_size=1024,
                                                   llm=llm,
                                                   embed_model=embeddings)
    # service_context = ServiceContext.from_defaults(llm=llm, 
    #                                                embed_model="local:/mnt/efs/data/saved_models/BAAI/bge-small-en-v1.5/model/")
    print('service_context created.\n')

    # Create the index
    index = VectorStoreIndex.from_documents(documents, #[document],
                                            service_context=service_context)
    print('index created.\n')
    return index, service_context


def load_mistral7b_query_engine(model, tokenizer, lst_docs, streaming=True):
    '''
    Return query_engine for given documents. 

    Args:
        model (transformers.models.mistral.modeling_mistral.MistralForCausalLM): Mistral 7B model (any size).
        tokenizer (transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast): Llama2 tokenizer.
        lst_doct (list): List of documents to be used in RAG model. 
        streaming (bool): True to set a streaming model, False otherwise. Defaults to True. 

    Returns:
        query_engine (type)
    
    TODO:
        Make embedding model and input. 
    '''
    # Read in PDF(s) into a llamaindex document object
    documents = SimpleDirectoryReader(input_files=lst_docs).load_data()
    
    # Print document stats
    print()
    print(f"Documents type: {type(documents)}")
    print(f"Number of pages: {len(documents)}")
    print(f"Sub-document type: {type(documents[0])}")
    print(f"{documents[0]}\n")
    print()
    
    # # Merge documents into one
    # document = Document(text="\n\n".join([doc.text for doc in documents]))

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

    # # Create a HF LLM using the index wrapper 
    # llm = HuggingFaceLLM(context_window=4096,
    #                      max_new_tokens=2048, #256,
    #                      system_prompt=system_prompt,
    #                      #  query_wrapper_prompt=query_wrapper_prompt,
    #                      model=model,
    #                      tokenizer=tokenizer, 
    #                      device_map='auto')
    # Create a HF LLM using the llama index wrapper 
    llm = HuggingFaceLLM(context_window=4096,
                        max_new_tokens=2048, #256,
                        system_prompt=system_prompt,
                        query_wrapper_prompt=PromptTemplate("<s>[INST] {query_str} [/INST] </s>\n"),
                        #  generate_kwargs={"temperature": 0.3, "do_sample": True},
                        generate_kwargs={"temperature": 0.2, "top_k": 5, "top_p": 0.95},
                        tokenizer_kwargs={"max_length": 4096},
                        model_kwargs={"quantization_config": NF4_CONFIG},
                        device_map="auto",
                        model=model,
                        tokenizer=tokenizer)

    # Create and dl embeddings instance  
    embeddings = LangchainEmbedding(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    print('embeddings loaded.\n')

    # Create new service context instance
    service_context = ServiceContext.from_defaults(chunk_size=1024,
                                                   llm=llm,
                                                   embed_model=embeddings)
    # service_context = ServiceContext.from_defaults(llm=llm, 
    #                                                embed_model="local:/mnt/efs/data/saved_models/BAAI/bge-small-en-v1.5/model/")
    print('service_context created.\n')

    # Create the index
    index = VectorStoreIndex.from_documents(documents, #[document],
                                            service_context=service_context)
    print('index created.\n')

    # Create the query engine
    query_engine = index.as_query_engine(streaming=streaming)
    print('query_engine loaded.\n')

    # Return 
    return query_engine


def load_mistral7b_CondenseQuestionChatEngine(model, tokenizer, lst_docs, streaming=True):
    '''
    Return query_engine for given documents. 

    Args:
        model (transformers.models.mistral.modeling_mistral.MistralForCausalLM): Mistral 7B model (any size).
        tokenizer (transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast): Llama2 tokenizer.
        lst_doct (list): List of documents to be used in RAG model. 
        streaming (bool): True to set a streaming model, False otherwise. Defaults to True. 

    Returns:
        query_engine (type)
    
    TODO:
        Make embedding model and input. 
    '''
    # Load in query engine
    query_engine = load_mistral7b_query_engine(model, tokenizer, lst_docs, streaming)

    # Create custom prompt
    custom_prompt = PromptTemplate("""\
        Given a conversation (between Human and Assistant) and a follow up message from Human, \
        rewrite the message to be a standalone question that captures all relevant context \
        from the conversation.

        <Chat History>
        {chat_history}

        <Follow Up Message>
        {question}

        <Standalone question>
        """)

    # Create chat_engine
    chat_engine = CondenseQuestionChatEngine.from_defaults(query_engine=query_engine,
                                                           condense_question_prompt=custom_prompt,
                                                           verbose=True)

    # Print 
    print('CondenseQuestionChatEngine loaded.\n')

    # Return 
    return chat_engine



# Load Mistral7B model saved to disk
def load_mistral8x7b_model(quantization=True,
                           save_dir='/mnt/efs/data/saved_models'):
    '''
    Loads Mixtral-8x7B-v0.1 model saved in mnt/efs/data/saved_models.

    Args:
        quantization (bool): True to use nf4_config. 
        save_dir (str): Path where models are saved.

    Returns:
        model (transformers.models.llama.modeling_llama.LlamaForCausalLM): Model.
        tokenizer (transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast): 
            Tokenizer.
    '''
    # Set model name 
    model_name = 'Mixtral-8x7B-v0.1'

    if torch.cuda.is_available():
        # Set model paths 
        path_model_save           = os.path.join(save_dir, model_name)
        path_model_save_model     = os.path.join(path_model_save, 'model')
        path_model_save_tokenizer = os.path.join(path_model_save, 'tokenizer')
        # Load model based on quantization
        if quantization:
            model = AutoModelForCausalLM.from_pretrained(path_model_save_model, quantization_config=NF4_CONFIG, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(path_model_save_model, load_in_4bit=False, device_map='auto')
        # Load tokernizer
        tokenizer = AutoTokenizer.from_pretrained(path_model_save_tokenizer) 
        # Return 
        return model, tokenizer