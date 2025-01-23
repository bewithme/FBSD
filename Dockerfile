FROM registry.cn-hangzhou.aliyuncs.com/bewithmeallmylife/11.4.0-cudnn8-runtime-ubuntu18.04-conda-python3.8-qt5:1.0.0

USER root
ENV PATH /root/anaconda3/bin:$PATH
RUN  conda create -n fbsd-classification python=3.10 -y
SHELL ["conda", "run", "-n", "fbsd-classification", "/bin/bash", "-c"]
ENV LANGUAGE zh_CN:zh


RUN apt update
RUN apt-get install ffmpeg
RUN pip install torch -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
RUN pip install transformers  -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
RUN pip install torchvision  -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
RUN pip uninstall numpy -y
RUN pip install numpy==1.26.4 -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com


ENV HF_ENDPOINT https://hf-mirror.com
ENV HF_HOME /hugging_face_home




WORKDIR /app/fbsd-classification

CMD ["/root/anaconda3/envs/vit-classification/bin/python","train.py"]

#sudo docker build -t='registry.cn-hangzhou.aliyuncs.com/bewithmeallmylife/vit-classification-cuda-11.4.0:fbsd-1.0.0' .

#sudo docker run --gpus all  -v /home/ubuntu/vit-classification:/app/vit-classification   -v /hugging_face_home:/hugging_face_home registry.cn-hangzhou.aliyuncs.com/bewithmeallmylife/vit-classification-cuda-11.4.0:fbsd-1.0.0
