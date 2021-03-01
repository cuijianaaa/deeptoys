# DeepToys
DeepToys is not a from scratch framework of deep learning. It uses high-quality and mature open source code as far as possible to help you solve the training and deployment of deep learning in one stop. It abstracts pipeline code that is often written repeatedly in different tasks to help you reduce repetitive work. For training we use [pytorch-1.0.0](https://pytorch.org/). DeepToys attaches great importance to deployment, focuses on embedded devices. For simple deployment, we simply encapsulate a class based on [ncnn](https://github.com/Tencent/ncnn).

## Features and Todos
The marked ones are now supported, and the rest are will be supported in the future
- [x] train/test/deploy by the pipeline with gpu or cpu
- [x] support training from scratch & finetune(only use the trained weights) & resume(use trained weights and optimizer settings, such as lr, momentum and etc.)
- [x] collect and manage losses and evaluation metrics by pipeline
- [x] log the losses and evaluation metrics and other training statistics to terminal & tensorboard
- [x] test and evaluate every n epochs
- [x] generate deploy model after training automatically(save onnx and convert it to ncnn model, and model.json used by deploy framework)
- [x] deploy code follows the [google c++ style](https://google.github.io/styleguide/cppguide.html), use [bazel](https://bazel.build/) to build
- [x] deploy code supports plugin to implement ncnn not supported operator
- [x] deploy use json file to define a operator or plugin run flow
- [ ] more example models
- [ ] support latest pytorch version
- [ ] deploy code supports cross compilation for embedded linux or android
## Download
```
mkdir ~/repos && cd ~/repos
git clone https://github.com/cuijianaaa/deeptoys.git
```
## Install and Setup

### virtualenv

```
# Install dependencies
sudo apt-get install python3-pip

# Install virtualenv
pip3 install virtualenv==16.0

# Create virtual python env
virtualenv ~/deeptoys_venv --python=python3.6
source ~/deeptoys_venv/bin/activate 

# Add a quick command 'dt' to alias to start working on DeepToys
echo "alias dt='source ~/deeptoys_venv/bin/activate && cd ~/repos/deeptoys'" >> ~/.bash_aliases
source ~/.bash_aliases

# Star work on deeptoys
dt

# Quit virtualenv only after you install all following packages
deactivate
```
### python requirements
```
# Should run under virtualenv
cd deeptoys
pip install -r requirements.txt
```
### pytorch

Refer to https://pytorch.org/ to download and install pytorch 1.0.0 and torchvision 0.2.0
```
# For example(my cuda version is low, maybe not suitable for your case..):
pip install https://download.pytorch.org/whl/cu80/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
pip install https://download.pytorch.org/whl/torchvision-0.2.0-py2.py3-none-any.whl
```

### ncnn
Refer to the following instructions or https://github.com/Tencent/ncnn to install ncnn
```
# Install dependencies
sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler libvulkan-dev vulkan-utils libopencv-dev

# Download code
cd ~/repos
git clone https://github.com/Tencent/ncnn.git

# Build and install
cd ncnn
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/ncnn -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON ..
make -j$(nproc)
sudo make install
```

After install ncnn, set onnx2ncnn tool path in pipeline.py, in Pipeline._load_default_pipe_cfg() set PIPELINE.DEPLOY.ONNX2NCNN = '/path/to/ncnn/build/tools/onnx/onnx2ncnn'

### deploy
Go to [deploy/](deploy/) and refer to [deploy/README.md](deploy/README.md) to build and test deploy code

## How to use
### Run example mnist:
```
# Train
cd example/classification/mnist
sudo chmod +x train.sh
./train.sh 0  # use gpu 0

# Deploy
# If ncnn is generated after training, no need to run deploy.sh
sudo chmod +x deploy.sh
./deploy.sh

# Run inference by deploy code
# 1. Copy deploy model to deploy folder
cp results/deploy/mnist* ../../../deploy/models/mnist/

# 2. Build deploy code
cd ../../../deploy
bazel build //src:mnist_main

# 3. Run inference
bazel-bin/src/mnist_main images/mnist.jpg

```
### Add a new task:
#### 1. Implement dataset loader
Inherit class Dataset in torch.utils.data.dataset
to implement your self dataset loader. Refer to [dataset/mnist.py](dataset/mnist.py), in this case we inherit the Inherited class MNIST for convenience
#### 2. Implement model and pipeline
Refer to [model/classification/mnist/](model/classification/mnist/) to add a new task folder, implement model and pipeline, in model/task/model_name/ write pipe.py to inherit class Pipeline,
override and implement loss, eval_step, eval functions
#### 3. Run train and deploy
#### 4. Copy deploy model
```
cp example/task/model_name/result/deploy/* deploy/models/model_name/
```
#### 5. Implement deploy code
##### Inherit and implement the Inference Engine

Refer to [deploy/src/mnist.cc](deploy/src/mnist.cc), [deploy/src/mnist.h](deploy/src/mnist.h) to implement a class to inherit class InferenceEngine, if there are some operators that ncnn doesn't support, you can write a plugin and call RegisterPlugin to register it, and write it in deploy/models/model_name/model_name.json

##### Write a main function to call

Refer to [deploy/src/mnist_main.cc](deploy/src/mnist_main.cc) to implement a main function to call the inference engine

#### 6. Build & Run
Refer to [deploy/src/BUILD](deploy/src/BUILD) to write the build rule for the new task

```
# Build
cd deploy
bazel build //src:model_name_main

# Run
bazel-bin/src/model_name_main
```