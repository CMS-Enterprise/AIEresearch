# TruLens Debug

## Backgound 
We want to emualte how [TruLens](https://www.trulens.org/) is used in the [DeepLeaning.AI RAG Class](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag) (see `/mnt/efs/data/AIEresearch/DeepLearningAI.RAGClass/L1-Advanced_RAG_Pipeline.ipynb`). 
```
ipy
tru = Tru()
tru.reset_database()
```

## Installation 
There are two TruLens installations. First is [trulens_explain](https://www.trulens.org/trulens_explain/install/). This is installed in `.venv_dev`. Alas, this is not what we are currently interested in. 
```
pip install trulens
```

The relavant installtion is for [trulens_eval](https://www.trulens.org/trulens_eval/install/). This is also installed in  `.venv_dev`. 
```
pip install trulens-eval
```

## Issue 1
When we import `truelens_eval` we get the error below (see `/mnt/efs/data/AIEresearch/demo_medicare_handbook/RAG/RAG.ipynb`).
```
Output exceeds the size limit. Open the full output data in a text editor---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
File /mnt/efs/data/AIEresearch/.venv_dev/lib/python3.9/site-packages/fastavro/read.py:2
      1 try:
----> 2     from . import _read
      3 except ImportError:

File fastavro/_read.pyx:10, in init fastavro._read()

File /usr/local/lib/python3.9/lzma.py:27
     26 import os
---> 27 from _lzma import *
     28 from _lzma import _encode_filter_properties, _decode_filter_properties

ModuleNotFoundError: No module named '_lzma'

During handling of the above exception, another exception occurred:

ModuleNotFoundError                       Traceback (most recent call last)
Cell In[2], line 1
----> 1 from trulens_eval import Tru

File /mnt/efs/data/AIEresearch/.venv_dev/lib/python3.9/site-packages/trulens_eval/__init__.py:83
      1 """
      2 # Trulens-eval LLM Evaluation Library
...
---> 27 from _lzma import *
     28 from _lzma import _encode_filter_properties, _decode_filter_properties
     29 import _compression

ModuleNotFoundError: No module named '_lzma'
```

> THIS NO LONGER SEEM NECESSARY 

Pull fix from yolo5 [ModuleNotFoundError: No module named '_lzma' #1298
](https://github.com/ultralytics/yolov5/issues/1298)
```
sudo yum install xz-devel
sudo yum install python-backports-lzma
pip install backports.lzma
```
Edit file `/usr/local/lib/python3.9/lzma.py:27` 
```
sudo chmod 777 /usr/local/lib/python3.9/lzma.py
```
lines 27 and 28 from 
```
---> 27 from _lzma import *
     28 from _lzma import _encode_filter_properties, _decode_filter_properties
```
to
```
try:
    from _lzma import *
    from _lzma import _encode_filter_properties, _decode_filter_properties
except:
    from backports.lzma import *
    from backports.lzma import _encode_filter_properties, _decode_filter_properties
```

> This issue seems to be resolved, but a new one popped up. 


## Issue 2

```
Output exceeds the size limit. Open the full output data in a text editor---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[1], line 1
----> 1 from trulens_eval import Tru

File /mnt/efs/data/AIEresearch/.venv_dev/lib/python3.9/site-packages/trulens_eval/__init__.py:83
      1 """
      2 # Trulens-eval LLM Evaluation Library
      3 
   (...)
     78 
     79 """
     81 __version__ = "0.18.3"
---> 83 from trulens_eval.feedback import Bedrock
     84 from trulens_eval.feedback import Feedback
     85 from trulens_eval.feedback import Huggingface

File /mnt/efs/data/AIEresearch/.venv_dev/lib/python3.9/site-packages/trulens_eval/feedback/__init__.py:16
     14 from trulens_eval.feedback.embeddings import Embeddings
     15 # Main class holding and running feedback functions:
---> 16 from trulens_eval.feedback.feedback import Feedback
     17 from trulens_eval.feedback.groundedness import Groundedness
     18 from trulens_eval.feedback.groundtruth import GroundTruthAgreement

File /mnt/efs/data/AIEresearch/.venv_dev/lib/python3.9/site-packages/trulens_eval/feedback/feedback.py:17
...
     71 ):
     72     if client is None:
     73         if client_kwargs is None and client_cls is None:

TypeError: unsupported operand type(s) for |: 'type' and 'type'
```

* Pull fix from https://stackoverflow.com/questions/76712720/typeerror-unsupported-operand-types-for-type-and-nonetype. Seems to be a feature that is supported in Python 3.10, while we are running 3.9. 


* To clean install python3.10+ you need to remove the old ssl and install anew.
```
cd /usr/bin/
sudo yum install gcc openssl-devel bzip2-devel libffi-devel zlib-devel -y 
sudo yum remove openssl-devel.x86_64
sudo yum remove openssl11-1.1.1g openssl11-libs-1.1.1g openssl11-devel-1.1.1g
sudo yum install openssl11-1.1.1g openssl11-libs-1.1.1g openssl11-devel-1.1.1g
```

* Install Python 3.12
```
sudo yum install gcc openssl-devel bzip2-devel libffi-devel zlib-devel -y 
cd /usr/bin/
sudo wget https://www.python.org/ftp/python/3.12.1/Python-3.12.1.tgz 
sudo tar xzf Python-3.12.1.tgz
cd Python-3.12.1 
sudo yum remove openssl-devel.x86_64
sudo yum remove openssl11-1.1.1g openssl11-libs-1.1.1g openssl11-devel-1.1.1g
sudo yum install openssl11-1.1.1g openssl11-libs-1.1.1g openssl11-devel-1.1.1g
sudo ./configure  --enable-optimizations
sudo make altinstall
python3.12 -V 
sudo rm -f /opt/Python-3.12.1.tgz 
```

* Unable to install torch, so try 3.11
```
sudo yum install gcc openssl-devel bzip2-devel libffi-devel zlib-devel -y 
cd /usr/bin/
sudo wget https://www.python.org/ftp/python/3.11.7/Python-3.11.7.tgz 
sudo tar xzf Python-3.11.7.tgz
cd Python-3.11.7 
sudo ./configure  --enable-optimizations
sudo make altinstall
python3.11 -V 
sudo rm -f /opt/Python-3.11.7 .tgz 
```
> This seems to be working. 


* python10
```
cd /usr/bin
sudo wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz 
sudo tar xzf Python-3.10.13.tgz
cd Python-3.10.13 
sudo ./configure  --enable-optimizations
sudo make altinstall
python3.10 -V 
sudo rm -f /opt/Python-3.10.13 .tgz 
```