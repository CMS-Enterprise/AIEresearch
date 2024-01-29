# VENV README

This document captures how Python Virtual Environments are used for the AIE Team. 

There is one venv for production `.venv_prod` and one for testing/development `.venv_dev`. 
In addition, there is a special `tmp` dir (`.venv_tmp`) used to handle space issues. 


## `.venv_tmp`
The temporary directory where pip install packages (`/tmp`) is small and needs to be changed. 
We use `./.venv_tmp` instead (for both `.venv_prod` and `.venv_dev`). 
Use `export TMPDIR=/mnt/efs/data/AIEresearch/.venv_tmp` before running pip commands per terminal session per venv.
This dir will have to cleared out from time to time. 
```
find .venv_tmp -path '*/*' -delete
```


## Install Python from binary
To clean install python3.10+ (change the version in the code below) you need to remove the old ssl and install anew (in that order!).
This is necessary on all new servers.
```
cd /usr/bin/
sudo yum install gcc openssl-devel bzip2-devel libffi-devel zlib-devel -y 
sudo yum remove  openssl openssl-devel.x86_64 -y
sudo yum remove  openssl11-1.1.1g openssl11-libs-1.1.1g openssl11-devel-1.1.1g -y
sudo yum install openssl11-1.1.1g openssl11-libs-1.1.1g openssl11-devel-1.1.1g -y
sudo yum install xz-devel -y

sudo wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz 
sudo tar xzf Python-3.10.13.tgz
cd Python-3.10.13 
sudo ./configure  --enable-optimizations
sudo make altinstall
python3.10 -V 
sudo rm -f /opt/Python-3.10.13 .tgz 
```


## `.venv_prod`
```
python3.9 -m venv .venv_prod
source .venv_prod/bin/activate 
pip install --upgrade pip
pip install ipykernel 
python -m ipykernel install --user --name .venv_prod
pip install torch 
pip install transformers
pip install gradio
pip install urllib3==1.26.0
pip install spaces
pip install bitsandbytes
pip install accelerate
pip install scipy
pip install pypdf
```
Need to install:
* llama_index
* langchain



## `.venv_dev`
```
python3.9 -m venv .venv_dev
source .venv_dev/bin/activate 
pip install --upgrade pip
pip install ipykernel 
python -m ipykernel install --user --name .venv_dev
pip install torch 
pip install transformers
pip install gradio
pip install urllib3==1.26.0
pip install spaces
pip install bitsandbytes
pip install accelerate
pip install scipy
pip install llama_index
pip install langchain
pip install pypdf

> pip install trulens
> pip install trulens-eval
```

Notes
* Check out https://www.giskard.ai/ instead of Trulens

## `.venv_dev310`
```
python3.10 -m venv .venv_dev310
pip install --upgrade pip
pip install ipykernel 
python -m ipykernel install --user --name .venv_dev310 # THIS ONE HAS TO BE DONE PER NEW SERVER
pip install torch transformers gradio
pip install bitsandbytes accelerate scipy
pip install llama_index langchain pypdf
pip install sentence-transformers nvidia-ml-py ipdb 
pip install trulens trulens-eval 
pip install jupyter --upgrade
pip install ipywidgets --upgrade
pip install networkx --upgrade
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
python -m pip install optimum[neuronx]
pip install transformers-neuronx --extra-index-url=https://pip.repos.neuron.amazonaws.com
python -m pip install git+https://github.com/aws-neuron/transformers-neuronx.git


pip install llama-cpp-python
```

Install git lfs
* https://stackoverflow.com/questions/71448559/git-large-file-storage-how-to-install-git-lfs-on-aws-ec2-linux-2-no-package
```
sudo amazon-linux-extras install epel -y 
sudo yum-config-manager --enable epel
sudo yum install git-lfs -y
```


## `.venv_dev311`
```
python3.11 -m venv .venv_dev311
source .venv_dev311/bin/activate 
pip install --upgrade pip
pip install ipykernel 
python -m ipykernel install --user --name .venv_dev311
pip install torch
pip install transformers
pip install gradio
pip install spaces
pip install bitsandbytes
pip install accelerate
pip install scipy
pip install llama_index
pip install langchain
pip install pypdf
pip install trulens
pip install trulens-eval
pip install sentence-transformers
pip install jupyter --upgrade
pip install ipywidgets --upgrade
pip install nvidia-ml-py
pip install ipdb
pip install --upgrade protobuf
pip install llama-cpp-python #
pip install dash
pip install dash_bootstrap_components


```


## `.venv_dev312`
```
python3.12 -m venv .venv_dev312
source .venv_dev312/bin/activate 
pip install --upgrade pip
pip install ipykernel 
python -m ipykernel install --user --name .venv_dev312
<!-- pip install torch  --> NOT WORKING
pip install transformers
pip install gradio
pip install spaces
pip install bitsandbytes
pip install accelerate
pip install scipy
pip install llama_index
pip install langchain
pip install pypdf
pip install trulens
<!-- pip install trulens-eval -->
```
Collecting pydantic-core==2.0.2 (from pydantic>=1.10.7->trulens-eval)
  Using cached pydantic_core-2.0.2.tar.gz (305 kB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... error
  error: subprocess-exited-with-error
  
  × Preparing metadata (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [6 lines of output]
      
      Cargo, the Rust package manager, is not installed or is not on PATH.
      This package requires Rust and Cargo to compile extensions. Install it through
      the system's package manager or via https://rustup.rs/
      
      Checking for Rust toolchain....
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
```