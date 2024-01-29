#!/bin/bash
cd /usr/bin/
sudo yum install gcc openssl-devel bzip2-devel libffi-devel zlib-devel -y 
sudo yum remove  openssl openssl-devel.x86_64 -y
sudo yum remove  openssl11-1.1.1g openssl11-libs-1.1.1g openssl11-devel-1.1.1g -y
sudo yum install openssl11-1.1.1g openssl11-libs-1.1.1g openssl11-devel-1.1.1g -y
sudo wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz 
sudo tar xzf Python-3.10.13.tgz
cd Python-3.10.13 
sudo ./configure  --enable-optimizations
sudo make altinstall
python3.10 -V 
sudo rm -f /opt/Python-3.10.13 .tgz 
cd /mnt/efs/data/AIEresearch
source .venv_dev310/bin/activate
python -m ipykernel install --user --name .venv_dev310 