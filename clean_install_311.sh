#!/bin/bash
cd /usr/bin/
sudo yum install gcc openssl-devel bzip2-devel libffi-devel zlib-devel -y 
sudo yum remove  openssl openssl-devel.x86_64 -y
sudo yum remove  openssl11-1.1.1g openssl11-libs-1.1.1g openssl11-devel-1.1.1g -y
sudo yum install openssl11-1.1.1g openssl11-libs-1.1.1g openssl11-devel-1.1.1g -y
sudo yum install xz-devel -y
sudo amazon-linux-extras install epel -y 
sudo yum-config-manager --enable epel
sudo yum install git-lfs -y

sudo wget https://www.python.org/ftp/python/3.11.7/Python-3.11.7.tgz 
sudo tar xzf Python-3.11.7.tgz
cd Python-3.11.7 
sudo ./configure  --enable-optimizations
sudo make altinstall
python3.11 -V 
sudo rm -f /opt/Python-3.11.7 .tgz 

cd /mnt/efs/data/AIEresearch
source .venv_dev311/bin/activate
python -m ipykernel install --user --name .venv_dev311 