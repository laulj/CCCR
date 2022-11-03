FROM paddlepaddle/paddle:2.1.1-gpu-cuda11.2-cudnn8

#RUN addgroup --system test
#RUN adduser --system testuser --ingroup test

#USER test:testuser

COPY . /home/PaddleOCR/

WORKDIR /home/PaddleOCR/

RUN \
	# Update nvidia GPG key
	rm /etc/apt/sources.list.d/cuda.list && \
	rm /etc/apt/sources.list.d/nvidia-ml.list && \
	apt-key del 7fa2af80 && \
	apt-get update && apt-get install -y --no-install-recommends wget && \
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
	dpkg -i cuda-keyring_1.0-1_all.deb && \
	apt-get update && \
 	apt-get install -y pciutils && \
 	python3 -m pip install --upgrade pip &&\
 	# Install the requirements
	pip3 install tensorflow==2.4.0 && \
	pip3 install --upgrade -r requirements.txt && \
	apt-get update
	
CMD ["/bin/bash", "cat", "verbose.txt"]
