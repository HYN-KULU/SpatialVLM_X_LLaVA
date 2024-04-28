FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
# ENV CUDA_HOME /usr/local/cuda-11.8/
# RUN apt-get update && apt-get install -y cmake
# RUN which cmake
# RUN cmake --version

# RUN apt-get update && apt-get install -y software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update && \
#     apt-get install -y python3.10 python3.10-dev python3.10-distutils curl

# # Set Python 3.10 as the default Python version
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
#     update-alternatives --set python3 /usr/bin/python3.10

# # Install pip using the get-pip.py script
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# # Install setuptools and wheel
# RUN python3.10 -m pip install setuptools wheel

# # Clean up to keep the image clean and compact
# RUN apt-get clean && \
#     rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# RUN apt-get update && \
# apt-get install --no-install-recommends -y wget ffmpeg libsm6 libxext6 git vim && \
# apt-get clean && \
# apt-get autoremove && \
# rm -rf /var/lib/apt/lists/*
    

# RUN apt-get install --no-install-recommends wget ffmpeg=7:* \
#     libsm6=2:* libxext6=2:* git=1:* vim=2:* -y \
#     && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

# RUN python3 -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# WORKDIR /app

# # Zoedepth dependencies
# RUN git clone https://github.com/isl-org/ZoeDepth.git

# # EfficientVit, SAM, llava
# RUN git clone https://github.com/mit-han-lab/efficientvit.git && cd efficientvit/ && pip3 install -r requirements.txt
# RUN wget https://huggingface.co/han-cai/efficientvit-sam/resolve/main/l0.pt

# ENV PATH="/usr/bin:/usr/local/bin:${PATH}"
# ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on" 
# RUN set -x \
#     && cmake --version \
#     && which cmake \
#     && cmake -E capabilities

# RUN ls -l /usr/local/bin/cmake  # Check permissions
# RUN chmod +x /usr/local/bin/cmake  # Make sure cmake is executable

# ##############################################################################################################

# RUN pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir --verbose

# RUN wget https://huggingface.co/cjpais/llava-v1.6-34B-gguf/resolve/main/mmproj-model-f16.gguf
# RUN wget https://huggingface.co/cjpais/llava-v1.6-34B-gguf/resolve/main/llava-v1.6-34b.Q4_K_M.gguf
# RUN python3 -m pip install segment-anything 

# # Additional dependencies
# RUN pip install open3d 
# RUN python3 -m pip install --no-cache-dir diffusers[torch]==0.15.1 opencv-python==4.7.0.72 \
#     pycocotools==2.0.6 matplotlib==3.5.3 \
#     onnxruntime==1.14.1 onnx==1.13.1 ipykernel==6.16.2 scipy gradio openai litellm

# RUN wget https://remyx.ai/assets/spatialvlm/warehouse_rgb.jpg

# COPY clipseg_data_processing.py /app
# ENTRYPOINT ["python3", "clipseg_data_processing.py"]
# CMD ["--input_image", "warehouse_rgb.jpg"]
